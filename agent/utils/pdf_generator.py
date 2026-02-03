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
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


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
    """Extrait les informations de checklist via LLM, avec validation + nettoyage robuste."""
    content: str = ""
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
        content = re.sub(r"^```json\\s*", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"^```\\s*", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"\\s*```\\s*$", "", content).strip()

        extracted = json.loads(content)
        if not isinstance(extracted, dict):
            return get_fallback_data()

        return validate_and_clean_extracted_data(extracted)
    except json.JSONDecodeError:
        return get_fallback_data()
    except Exception:
        return get_fallback_data()


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
    text = re.sub(r"\\*\\*", "", text)
    text = re.sub(r"^[-•]\\s*", "", text.strip())
    text = re.sub(r"^\\d+\\.?\\s*", "", text)

    # Collapse whitespace.
    text = re.sub(r"\\s+", " ", text).strip()

    # Remove broken parentheses fragments.
    text = re.sub(r"\\(\\s*\\)", "", text)
    text = re.sub(r"\\($", "", text).strip()
    text = re.sub(r"^\\)", "", text).strip()

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
    value = re.sub(r"[^\\w\\s-]", "", value, flags=re.UNICODE).strip()
    value = re.sub(r"\\s+", "_", value)
    return value[:40] or "diagnostic"


def _escape_pdf_text(value: str) -> str:
    # ReportLab Paragraph uses a minimal XML-like markup.
    value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return value


class ChecklistPDFGenerator:
    """Génère des PDF de checklist diagnostic BTP (design professionnel, sans glyphes exotiques)."""

    def __init__(self):
        self.color_primary = colors.HexColor("#0066cc")
        self.color_secondary = colors.HexColor("#f0f7ff")
        self.color_alert = colors.HexColor("#ef4444")
        self.color_text = colors.HexColor("#1f2937")
        self.color_gray = colors.HexColor("#6b7280")

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
                    fontSize=11,
                    textColor=self.color_gray,
                    spaceAfter=22,
                    alignment=TA_CENTER,
                    fontName="Helvetica",
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
                    fontSize=10,
                    textColor=self.color_text,
                    spaceAfter=6,
                    alignment=TA_LEFT,
                    fontName="Helvetica",
                    leading=14,
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
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            leftMargin=2.5 * cm,
            rightMargin=2.5 * cm,
        )

        content: list = []

        # Header
        content.append(Paragraph("NEXTMIND", self.styles["CustomTitle"]))
        content.append(Paragraph("Checklist Diagnostic BTP", self.styles["CustomSubtitle"]))
        content.append(Spacer(1, 0.2 * cm))

        # Project info block
        info_data = [
            ["Projet :", _escape_pdf_text(project_name)],
            ["Date :", datetime.now().strftime("%d/%m/%Y à %H:%M")],
        ]
        info_table = Table(info_data, colWidths=[3.5 * cm, 12.5 * cm])
        info_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("TEXTCOLOR", (0, 0), (0, -1), self.color_gray),
                    ("TEXTCOLOR", (1, 0), (1, -1), self.color_text),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        content.append(info_table)
        content.append(Spacer(1, 0.7 * cm))

        if checkpoints:
            content.append(Paragraph("1. POINTS DE CONTRÔLE PRIORITAIRES", self.styles["SectionHeader"]))
            content.append(Spacer(1, 0.25 * cm))
            content.extend(self._checkbox_list(checkpoints, index_prefix=True))
            content.append(Spacer(1, 0.5 * cm))

        if alerts:
            content.append(Paragraph("2. SIGNAUX D'ALERTE", self.styles["SectionHeader"]))
            content.append(Spacer(1, 0.25 * cm))
            content.extend(self._alert_list(alerts))
            content.append(Spacer(1, 0.5 * cm))

        if photos_needed:
            content.append(Paragraph("3. PHOTOS À PRENDRE", self.styles["SectionHeader"]))
            content.append(Spacer(1, 0.25 * cm))
            content.extend(self._photo_list(photos_needed))
            content.append(Spacer(1, 0.5 * cm))

        if materials:
            content.append(Paragraph("4. MATÉRIAUX NÉCESSAIRES", self.styles["SectionHeader"]))
            content.append(Spacer(1, 0.25 * cm))
            content.extend(self._bullet_list(materials))
            content.append(Spacer(1, 0.5 * cm))

        footer_style = ParagraphStyle(
            "Footer",
            parent=self.styles["Normal"],
            fontSize=9,
            textColor=self.color_gray,
            alignment=TA_CENTER,
            fontName="Helvetica-Oblique",
        )
        content.append(Spacer(1, 0.8 * cm))
        content.append(Paragraph("Document généré par NEXTMIND - Assistant IA BTP professionnel", footer_style))

        doc.build(content)
        return output_path

    def _checkbox_list(self, items: list[str], *, index_prefix: bool) -> list:
        rows: list[list] = []
        for idx, item in enumerate(items, 1):
            label = f"{idx}. {item}" if index_prefix else item
            rows.append([Paragraph("", self.styles["CustomBody"]), Paragraph(_escape_pdf_text(label), self.styles["CustomBody"])])

        table = Table(rows, colWidths=[0.55 * cm, 15.45 * cm], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("GRID", (0, 0), (0, -1), 0.9, self.color_primary),
                ]
            )
        )
        return [table]

    def _alert_list(self, items: list[str]) -> list:
        rows: list[list] = []
        for item in items:
            icon = Paragraph("<font color='#ef4444'><b>!</b></font>", self.styles["CustomBody"])
            rows.append([icon, Paragraph(_escape_pdf_text(item), self.styles["CustomBody"])])

        table = Table(rows, colWidths=[0.55 * cm, 15.45 * cm], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return [table]

    def _photo_list(self, items: list[str]) -> list:
        rows: list[list] = []
        for idx, item in enumerate(items, 1):
            icon = Paragraph("<font color='#6b7280'>Photo</font>", self.styles["CustomBody"])
            rows.append([icon, Paragraph(_escape_pdf_text(f"{idx}. {item}"), self.styles["CustomBody"])])

        table = Table(rows, colWidths=[0.9 * cm, 15.1 * cm], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return [table]

    def _bullet_list(self, items: list[str]) -> list:
        bullets: list = []
        for item in items:
            bullets.append(Paragraph(f"• {_escape_pdf_text(item)}", self.styles["CustomBody"]))
        return bullets
