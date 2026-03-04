from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


CHECKLIST_EXTRACTION_PROMPT = """
Tu es un extracteur d'informations techniques BTP. Ton rôle est d'analyser la réponse d'un assistant et d'en extraire UNIQUEMENT les informations pertinentes pour une checklist de diagnostic.

Réponse de l'assistant à analyser :
{response_text}

RÈGLES D'EXTRACTION STRICTES :

1. POINTS DE CONTRÔLE :
   - Ce sont les vérifications à faire sur site
   - Format : "Vérifier X", "Contrôler Y", "Inspecter Z"
   - NE PAS inclure : prix, durées, conseils généraux
   - Exemples valides : "Vérifier la largeur des fissures avec pied à coulisse", "Contrôler l'état du support"
   - Max 6 items

2. SIGNAUX D'ALERTE :
   - Ce sont les indicateurs de problèmes graves/urgents
   - Format : "Si X alors risque de Y"
   - NE PAS inclure : bonnes pratiques générales
   - Exemples valides : "Fissures > 1cm = risque structurel", "Support humide = traiter avant réparation"
   - Max 4 items

3. PHOTOS À PRENDRE :
   - Description précise de quoi photographier et depuis quel angle
   - Format : "Photo de X (vue Y)"
   - Exemples valides : "Vue d'ensemble de la façade (recul 5-10m)", "Zoom sur chaque fissure avec règle graduée visible"
   - Max 5 items

4. MATÉRIAUX NÉCESSAIRES :
   - Liste des matériaux avec quantités si mentionnées
   - Format : "Matériau : quantité approximative"
   - Exemples valides : "Enduit de rebouchage : 2-3 seaux 5kg", "Primaire d'accrochage : 1L"
   - Max 6 items

RÉPONDS UNIQUEMENT EN JSON STRICT (pas de texte avant/après, pas de markdown) :

{{
  "project_name": "Titre court du diagnostic (ex: Diagnostic fissures murs)",
  "checkpoints": [
    "Point de contrôle 1 (action à faire sur site)",
    "Point de contrôle 2"
  ],
  "alerts": [
    "Signal d'alerte 1 (condition = risque)",
    "Signal d'alerte 2"
  ],
  "photos": [
    "Photo 1 : description + angle de vue",
    "Photo 2 : ..."
  ],
  "materials": [
    "Matériau 1 : quantité",
    "Matériau 2 : ..."
  ]
}}

IMPORTANT :
- Pas de caractères parasites (**, ■, ■■)
- Pas de texte coupé
- Pas de numérotation dans les items (le PDF les numérotera)
- Si une catégorie est absente de la réponse : liste vide []
- Items concis (< 120 caractères)
""".strip()


@dataclass(frozen=True)
class ExtractedChecklist:
    project_name: str
    checkpoints: list[str]
    alerts: list[str]
    photos: list[str]
    materials: list[str]


def extract_checklist_info_with_llm(response_text: str) -> dict[str, Any]:
    """Extrait une checklist à partir d'un texte d'agent.

    Priorité:
    1) extraction LLM (si OPENAI_API_KEY + lib openai dispo)
    2) extraction heuristique (robuste, sans dépendances externes)
    3) fallback statique (dernier recours)
    """

    response_text = (response_text or "").strip()
    if not response_text:
        return get_fallback_data()

    def heuristic() -> dict[str, Any]:
        return validate_and_clean_extracted_data(extract_checklist_info_heuristic(response_text))

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return heuristic()

    content: str = ""
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("CHECKLIST_PDF_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": CHECKLIST_EXTRACTION_PROMPT.format(response_text=response_text),
                }
            ],
            temperature=0,
            max_tokens=1000,
        )
        content = (response.choices[0].message.content or "").strip()

        # Remove accidental code fences, if any.
        content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"^```\s*", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"\s*```\s*$", "", content).strip()

        extracted = json.loads(content)
        if not isinstance(extracted, dict):
            return heuristic()

        return validate_and_clean_extracted_data(extracted)
    except json.JSONDecodeError:
        return heuristic()
    except Exception:
        return heuristic()


def _guess_project_name_from_text(text: str) -> str:
    lines = [clean_text(x) for x in (text or "").splitlines()]
    lines = [x for x in lines if x]

    for line in lines[:10]:
        # e.g. "Checklist Préparation Démolition de Bâtiment :"
        m = re.match(r"^(?:check\s*-?\s*list|checklist)\s*(.+)$", line, flags=re.IGNORECASE)
        if m:
            value = m.group(1).strip().lstrip(":").strip()
            value = re.sub(r"\s*[:\-–—]+\s*$", "", value).strip()
            if value:
                return value[:60]

    return "Diagnostic BTP"


def _is_heading(line: str) -> bool:
    if not line:
        return False
    if re.match(r"^\d+\.", line):
        return True
    if line.endswith(":") and len(line) <= 80:
        return True
    return False


def _clean_item_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[-*•]+\s*", "", line)
    line = re.sub(r"^\d+[.)]\s*", "", line)
    return clean_text(line)


def _category_from_heading(heading: str) -> str:
    h = heading.lower()
    if "photo" in h:
        return "photos"
    if "matériau" in h or "materiau" in h or "epi" in h or "équipement" in h or "equipement" in h:
        return "materials"
    if "alerte" in h or "danger" in h:
        return "alerts"
    return "checkpoints"


def extract_checklist_info_heuristic(response_text: str) -> dict[str, Any]:
    """Extraction sans LLM: transforme une réponse structurée en listes exploitables.

    Supporte:
    - titres type "Checklist X :"
    - sections "1. ..." / "... :" (avec sous-lignes)
    - puces "-" "*" "•" ou listes simples ligne par ligne
    """

    project_name = _guess_project_name_from_text(response_text)

    checkpoints: list[str] = []
    alerts: list[str] = []
    photos: list[str] = []
    materials: list[str] = []

    current_heading = ""
    current_category = "checkpoints"

    lines = [x.rstrip() for x in (response_text or "").splitlines()]
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Ignore obvious intro lines
        if re.match(r"^checklist\b", line, flags=re.IGNORECASE):
            continue

        if _is_heading(line):
            current_heading = _clean_item_line(line.rstrip(":"))
            current_category = _category_from_heading(current_heading)
            continue

        # Skip non-item paragraphs that introduce a list (ending with ":")
        if line.endswith(":"):
            continue

        item = _clean_item_line(line)
        if not item or len(item) < 3:
            continue

        # Some items carry category in the content even if the heading doesn't.
        lowered = item.lower()
        if "photo" in lowered:
            photos.append(item.replace("Photo", "").strip(" :.-"))
            continue

        if any(token in lowered for token in ["casque", "gants", "lunettes", "masque", "epi", "barrière", "barriere"]):
            materials.append(item)
            continue

        if current_category == "alerts":
            alerts.append(item)
        elif current_category == "photos":
            photos.append(item)
        elif current_category == "materials":
            materials.append(item)
        else:
            checkpoints.append(item)

    # If no headings were detected, treat bullet-like lines as checkpoints.
    if not any([checkpoints, alerts, photos, materials]):
        for raw in lines:
            item = _clean_item_line(raw)
            if item:
                checkpoints.append(item)

    # Dedupe while keeping order
    def dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in items:
            key = x.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(x)
        return out

    checkpoints = dedupe(checkpoints)
    alerts = dedupe(alerts)
    photos = dedupe(photos)
    materials = dedupe(materials)

    return {
        "project_name": project_name,
        "checkpoints": checkpoints,
        "alerts": alerts,
        "photos": photos,
        "materials": materials,
    }


def validate_and_clean_extracted_data(data: dict[str, Any]) -> dict[str, Any]:
    required_defaults: dict[str, Any] = {
        "project_name": "Diagnostic BTP",
        "checkpoints": [],
        "alerts": [],
        "photos": [],
        "materials": [],
    }
    for key, default in required_defaults.items():
        if key not in data:
            data[key] = default

    data["project_name"] = clean_text(str(data.get("project_name") or "Diagnostic BTP"))[:60]
    data["checkpoints"] = clean_list_items(data.get("checkpoints"), max_items=6)
    data["alerts"] = clean_list_items(data.get("alerts"), max_items=4)
    data["photos"] = clean_list_items(data.get("photos"), max_items=5)
    data["materials"] = clean_list_items(data.get("materials"), max_items=6)

    return data


def clean_list_items(items: Any, *, max_items: int) -> list[str]:
    if not isinstance(items, list):
        return []

    cleaned: list[str] = []
    for item in items[:max_items]:
        if not isinstance(item, str):
            continue
        cleaned_item = clean_text(item)
        if not cleaned_item:
            continue
        cleaned.append(_truncate(cleaned_item, 120))

    return cleaned


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Remove common "parasite" glyphs and markdown.
    text = re.sub(r"[■●◆▪▫]", "", text)
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"^[-•]\s*", "", text.strip())
    text = re.sub(r"^\d+\.?\s*", "", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    # Remove broken parentheses fragments.
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\($", "", text).strip()
    text = re.sub(r"^\)", "", text).strip()

    return text


def get_fallback_data() -> dict[str, Any]:
    return {
        "project_name": "Diagnostic BTP",
        "checkpoints": [
            "Effectuer une inspection visuelle complète",
            "Prendre des mesures précises",
            "Documenter avec photos",
            "Vérifier l'état général du support",
        ],
        "alerts": [
            "Consulter un professionnel si doute sur la gravité",
            "Ne pas intervenir en cas de risque structurel apparent",
        ],
        "photos": [
            "Vue d'ensemble de la zone concernée",
            "Photos de détail des points problématiques",
        ],
        "materials": [],
    }


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _safe_filename(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE).strip()
    value = re.sub(r"\s+", "_", value)
    return value[:40] or "diagnostic"


def _escape_pdf_text(value: str) -> str:
    # ReportLab Paragraph uses a minimal XML-like markup.
    value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return value


class ChecklistPDFGenerator:
    """Génère des PDF de checklist diagnostic BTP (design professionnel, sans glyphes exotiques)."""

    def __init__(self):
        self.palette = {
            "primaryBlue": colors.HexColor("#0066CC"),
            "darkBlue": colors.HexColor("#004999"),
            "lightBlue": colors.HexColor("#E8F4FF"),
            "darkGray": colors.HexColor("#2C3E50"),
            "mediumGray": colors.HexColor("#7F8C8D"),
            "lightGray": colors.HexColor("#F8F9FA"),
            "warningRed": colors.HexColor("#E74C3C"),
            "warningBg": colors.HexColor("#FFEBEE"),
            "borderLight": colors.HexColor("#E0E0E0"),
            "white": colors.white,
        }

        # Backward-compatible aliases used by existing styles/helpers.
        self.color_primary = self.palette["primaryBlue"]
        self.color_secondary = self.palette["lightBlue"]
        self.color_alert = self.palette["warningRed"]
        self.color_text = self.palette["darkGray"]
        self.color_gray = self.palette["mediumGray"]

        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self) -> None:
        if "CustomTitle" not in self.styles:
            self.styles.add(
                ParagraphStyle(
                    name="CustomTitle",
                    parent=self.styles["Heading1"],
                    fontSize=22,
                    textColor=self.color_primary,
                    spaceAfter=8,
                    spaceBefore=0,
                    alignment=TA_CENTER,
                    fontName="Helvetica-Bold",
                )
            )

        if "CustomSubtitle" not in self.styles:
            self.styles.add(
                ParagraphStyle(
                    name="CustomSubtitle",
                    parent=self.styles["Normal"],
                    fontSize=12,
                    textColor=self.color_gray,
                    spaceAfter=14,
                    alignment=TA_CENTER,
                    fontName="Helvetica",
                )
            )

        if "SectionHeaderText" not in self.styles:
            self.styles.add(
                ParagraphStyle(
                    name="SectionHeaderText",
                    parent=self.styles["Normal"],
                    fontSize=13,
                    textColor=colors.white,
                    fontName="Helvetica-Bold",
                    leading=18,
                )
            )

        if "SectionHeader" not in self.styles:
            self.styles.add(
                ParagraphStyle(
                    name="SectionHeader",
                    parent=self.styles["Heading2"],
                    fontSize=13,
                    textColor=colors.white,
                    spaceAfter=0,
                    spaceBefore=14,
                    fontName="Helvetica-Bold",
                    backColor=self.color_primary,
                    leftIndent=8,
                    rightIndent=8,
                    leading=18,
                )
            )

        if "CustomBody" not in self.styles:
            self.styles.add(
                ParagraphStyle(
                    name="CustomBody",
                    parent=self.styles["BodyText"],
                    fontSize=11,
                    textColor=self.color_text,
                    spaceAfter=0,
                    alignment=TA_LEFT,
                    fontName="Helvetica",
                    leading=15,
                )
            )

        if "ListItem" not in self.styles:
            self.styles.add(
                ParagraphStyle(
                    name="ListItem",
                    parent=self.styles["BodyText"],
                    fontSize=10,
                    textColor=self.color_text,
                    spaceAfter=4,
                    leftIndent=0,
                    fontName="Helvetica",
                    leading=13,
                )
            )

        if "FooterPrimary" not in self.styles:
            self.styles.add(
                ParagraphStyle(
                    name="FooterPrimary",
                    parent=self.styles["Normal"],
                    fontSize=10,
                    textColor=self.color_primary,
                    alignment=TA_CENTER,
                    fontName="Helvetica-Bold",
                )
            )

        if "FooterSecondary" not in self.styles:
            self.styles.add(
                ParagraphStyle(
                    name="FooterSecondary",
                    parent=self.styles["Normal"],
                    fontSize=9,
                    textColor=self.color_gray,
                    alignment=TA_CENTER,
                    fontName="Helvetica-Oblique",
                )
            )

    def _find_logo_path(self) -> Path | None:
        env_path = os.getenv("NEXTMIND_LOGO_PATH")
        candidates: list[Path] = []
        if env_path:
            candidates.append(Path(env_path))

        candidates.extend(
            [
                Path.cwd() / "public" / "images" / "nextmind.png",
                Path.cwd().parent / "NextMind-main" / "public" / "images" / "nextmind.png",
                Path(__file__).resolve().parents[2] / "public" / "images" / "nextmind.png",
            ]
        )

        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate
            except Exception:
                continue
        return None

    def _logo_flowable(self, *, max_content_width: float) -> Image | None:
        logo_path = self._find_logo_path()
        if not logo_path:
            return None

        img = Image(str(logo_path))
        img.hAlign = "CENTER"

        # Keep header compact to avoid pushing the checklist to page 2.
        max_width = min(max_content_width, 6.0 * cm)
        max_height = 1.2 * cm

        try:
            img_w = float(getattr(img, "imageWidth", 0) or 0)
            img_h = float(getattr(img, "imageHeight", 0) or 0)
            if img_w > 0 and img_h > 0:
                scale = min(max_width / img_w, max_height / img_h, 1.0)
                img.drawWidth = img_w * scale
                img.drawHeight = img_h * scale
            else:
                img.drawWidth = max_width
        except Exception:
            img.drawWidth = max_width

        return img

    def _separator_line(self, width: float) -> Table:
        line = Table([[""]], colWidths=[width], rowHeights=[0.2 * cm])
        line.setStyle(
            TableStyle(
                [
                    ("LINEBELOW", (0, 0), (-1, -1), 6, self.color_primary),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return line

    def _project_info_table(self, *, width: float, project_name: str, date_text: str) -> Table:
        label_style = ParagraphStyle(
            "ProjectInfoLabel",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=self.color_primary,
            leading=13,
        )
        value_style = ParagraphStyle(
            "ProjectInfoValue",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=12,
            textColor=self.color_text,
            leading=15,
        )

        rows = [
            [Paragraph("PROJET", label_style), Paragraph(_escape_pdf_text(project_name), value_style)],
            [Paragraph("DATE", label_style), Paragraph(_escape_pdf_text(date_text), value_style)],
        ]

        label_width = 4.0 * cm
        value_width = max(0.0, width - label_width)
        table = Table(rows, colWidths=[label_width, value_width], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.6, self.palette["borderLight"]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        return table

    def _section_header_table(self, *, width: float, title: str, number: int, kind: str) -> Table:
        fill = self.palette["warningRed"] if kind == "alerte" else self.color_primary
        paragraph = Paragraph(f"{number}. {_escape_pdf_text(title.upper())}", self.styles["SectionHeaderText"])
        table = Table([[paragraph]], colWidths=[width])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), fill),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        return table

    def _items_table(self, *, width: float, items: list[str], kind: str) -> Table:
        rows: list[list[Any]] = []

        blue = self.palette["primaryBlue"].hexval()[2:]
        red = self.palette["warningRed"].hexval()[2:]
        dark = self.palette["darkGray"].hexval()[2:]

        for idx, raw in enumerate(items):
            item = _escape_pdf_text(raw)
            if kind == "checklist":
                prefix = f"<font color='#{blue}'><b>&#8226;</b></font> "
                paragraph = Paragraph(prefix + item, self.styles["CustomBody"])
            elif kind == "alerte":
                prefix = f"<font color='#{red}'><b>!</b></font> "
                paragraph = Paragraph(prefix + item, self.styles["CustomBody"])
            elif kind == "photos":
                prefix = f"<font color='#{dark}'><b>Photo</b></font> "
                paragraph = Paragraph(prefix + f"{idx + 1}. " + item, self.styles["CustomBody"])
            else:
                paragraph = Paragraph(item, self.styles["CustomBody"])

            rows.append([paragraph])

        table = Table(rows, colWidths=[width], hAlign="LEFT")
        style_cmds: list[tuple] = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]

        if kind == "alerte":
            style_cmds.append(("BACKGROUND", (0, 0), (-1, -1), self.palette["warningBg"]))
            style_cmds.append(("BOTTOMPADDING", (0, 0), (-1, -1), 10))
        else:
            for i in range(len(rows)):
                fill = self.palette["lightBlue"] if i % 2 == 1 else self.palette["white"]
                style_cmds.append(("BACKGROUND", (0, i), (0, i), fill))

        table.setStyle(TableStyle(style_cmds))
        return table

    def _footer_block(self, width: float) -> list[Any]:
        line = Table([[""]], colWidths=[width], rowHeights=[0.15 * cm])
        line.setStyle(
            TableStyle(
                [
                    ("LINEABOVE", (0, 0), (-1, -1), 0.6, self.palette["borderLight"]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        return [
            Spacer(1, 0.6 * cm),
            line,
            Spacer(1, 0.25 * cm),
            Paragraph("Document généré par NEXTMIND", self.styles["FooterPrimary"]),
            Spacer(1, 0.08 * cm),
            Paragraph("Assistant IA BTP professionnel", self.styles["FooterSecondary"]),
        ]

    def generate_checklist_pdf(
        self,
        *,
        project_name: str,
        checkpoints: list[str],
        alerts: list[str],
        photos_needed: list[str],
        materials: list[str] | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path.cwd() / "output" / f"checklist_{_safe_filename(project_name)}_{timestamp}.pdf"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            topMargin=2.54 * cm,
            bottomMargin=2.54 * cm,
            leftMargin=2.54 * cm,
            rightMargin=2.54 * cm,
        )

        content: list = []

        # Header (logo if available)
        logo = self._logo_flowable(max_content_width=doc.width)
        if logo:
            content.append(logo)
            content.append(Spacer(1, 0.18 * cm))
        else:
            content.append(Paragraph("NEXTMIND", self.styles["CustomTitle"]))

        content.append(Paragraph("Checklist Diagnostic BTP", self.styles["CustomSubtitle"]))
        content.append(self._separator_line(doc.width))
        content.append(Spacer(1, 0.25 * cm))

        date_text = datetime.now().strftime("%d/%m/%Y à %H:%M")
        content.append(self._project_info_table(width=doc.width, project_name=project_name, date_text=date_text))
        content.append(Spacer(1, 0.7 * cm))

        section_no = 1
        if checkpoints:
            content.append(
                self._section_header_table(
                    width=doc.width,
                    title="Points de contrôle prioritaires",
                    number=section_no,
                    kind="checklist",
                )
            )
            content.append(self._items_table(width=doc.width, items=checkpoints, kind="checklist"))
            content.append(Spacer(1, 0.55 * cm))
            section_no += 1

        if alerts:
            content.append(
                self._section_header_table(width=doc.width, title="Signaux d'alerte", number=section_no, kind="alerte")
            )
            content.append(self._items_table(width=doc.width, items=alerts, kind="alerte"))
            content.append(Spacer(1, 0.55 * cm))
            section_no += 1

        if photos_needed:
            content.append(
                self._section_header_table(width=doc.width, title="Photos à prendre", number=section_no, kind="photos")
            )
            content.append(self._items_table(width=doc.width, items=photos_needed, kind="photos"))
            content.append(Spacer(1, 0.55 * cm))
            section_no += 1

        if materials:
            content.append(
                self._section_header_table(
                    width=doc.width,
                    title="Matériaux nécessaires",
                    number=section_no,
                    kind="checklist",
                )
            )
            content.append(self._items_table(width=doc.width, items=materials, kind="checklist"))
            content.append(Spacer(1, 0.55 * cm))

        content.extend(self._footer_block(doc.width))

        doc.build(content)
        return output_path
