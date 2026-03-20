import logging
import asyncio
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Set

try:
    from agent.supabase_client import get_client
except ImportError:
    # Fallback or proper package path
    from supabase_client import get_client

logger = logging.getLogger(__name__)

TAG_MAPPING = {
    "plomberie": "plomberie",
    "electricite": "electricite", 
    "électricité": "electricite",
    "peinture": "peinture",
    "carrelage": "carrelage",
    "menuiserie": "menuiserie",
    "isolation": "isolation",
    "gros_oeuvre": "gros_oeuvre",
    "gros oeuvre": "gros_oeuvre",
    "maçonnerie": "maconnerie",
    "maconnerie": "maconnerie",
    "chauffage": "chauffage",
    "climatisation": "climatisation",
    "ventilation": "ventilation",
    "charpente": "charpente",
    "couverture": "couverture",
    "demolition": "demolition",
    "démolition": "demolition",
    "terrassement": "terrassement",
    "ravalement": "ravalement",
    "etancheite": "etancheite",
    "étanchéité": "etancheite",
}

class ProTagScorer:
    def __init__(self):
        self.supabase = get_client()
        # Alias conservé pour suivre la structure attendue par le pipeline.
        self.sb = self.supabase

    def _normalize_tag(self, lot_type: str, lot_name: str) -> Optional[str]:
        if lot_type:
            lt = str(lot_type).strip().lower()
            if lt in TAG_MAPPING:
                return TAG_MAPPING[lt]
            
        if lot_name:
            ln = str(lot_name).strip().lower()
            for key, val in TAG_MAPPING.items():
                if key in ln:
                    return val
        return None

    def _get_lots_for_pro(self, pro_id: str) -> List[dict]:
        """
        Retourne tous les lots liés à un pro avec un facteur de confiance.
        Structure retournée : [{...lot_fields, "confidence_factor": float}]
        """
        sb = self.sb
        if not sb or not pro_id:
            return []

        all_lots: List[dict] = []
        seen_lot_ids: Set[str] = set()

        # Source C : lots directs (responsible_user_id) -> factor 1.0
        direct_added = 0
        try:
            direct_lots = (
                sb.table("lots")
                .select("id,lot_type,name,status,delay_days")
                .eq("responsible_user_id", pro_id)
                .in_("status", ["en_cours", "termine", "valide"])
                .execute()
                .data
                or []
            )
            for lot in direct_lots:
                if not lot or lot.get("id") in seen_lot_ids:
                    continue
                lot["confidence_factor"] = 1.0
                all_lots.append(lot)
                seen_lot_ids.add(lot["id"])
                direct_added += 1
        except Exception as exc:
            logger.error("SOURCE C lots directs failed for pro_id=%s: %s", pro_id, exc, exc_info=True)

        # Source B : phases managées -> factor 0.8
        managed_phases = (
            sb.table("phases")
            .select("id")
            .eq("phase_manager_id", pro_id)
            .execute()
            .data
            or []
        )

        phase_added = 0
        if managed_phases:
            phase_ids = [p["id"] for p in managed_phases if p and p.get("id")]
            if phase_ids:
                try:
                    phase_lots = (
                        sb.table("lots")
                        .select("id,lot_type,name,status,delay_days")
                        .in_("phase_id", phase_ids)
                        .in_("status", ["en_cours", "termine", "valide"])
                        .execute()
                        .data
                        or []
                    )
                    for lot in phase_lots:
                        if lot and lot.get("id") not in seen_lot_ids:
                            lot["confidence_factor"] = 0.8
                            all_lots.append(lot)
                            seen_lot_ids.add(lot["id"])
                            phase_added += 1
                except Exception as exc:
                    logger.error("SOURCE B phases managées failed for pro_id=%s: %s", pro_id, exc, exc_info=True)

        # Source A : chef de projet -> factor 0.6
        managed_projects = (
            sb.table("projects")
            .select("id")
            .eq("project_manager_id", pro_id)
            .execute()
            .data
            or []
        )

        created_projects = (
            sb.table("projects")
            .select("id")
            .eq("created_by", pro_id)
            .execute()
            .data
            or []
        )

        member_projects = (
            sb.table("project_members")
            .select("project_id")
            .eq("user_id", pro_id)
            .in_("status", ["accepted", "active"])
            .in_(
                "role",
                [
                    "owner",
                    "collaborator",
                    "collaborateur",
                    "pro",
                    "professionnel",
                    "chef de projet",
                    "chef_de_projet",
                    "project manager",
                    "project_manager",
                ],
            )
            .execute()
            .data
            or []
        )

        project_added = 0
        all_project_ids_set: Set[str] = set()
        for p in (managed_projects + created_projects):
            if p and p.get("id"):
                all_project_ids_set.add(p["id"])
        for p in member_projects:
            if p and p.get("project_id"):
                all_project_ids_set.add(p["project_id"])
        all_project_ids = list(all_project_ids_set)

        if all_project_ids:
            try:
                project_phases = (
                    sb.table("phases")
                    .select("id")
                    .in_("project_id", all_project_ids)
                    .execute()
                    .data
                    or []
                )
                if project_phases:
                    project_phase_ids = [p["id"] for p in project_phases if p and p.get("id")]
                    if project_phase_ids:
                        project_lots = (
                            sb.table("lots")
                            .select("id,lot_type,name,status,delay_days")
                            .in_("phase_id", project_phase_ids)
                            .in_("status", ["en_cours", "termine", "valide"])
                            .execute()
                            .data
                            or []
                        )
                        for lot in project_lots:
                            if lot and lot.get("id") not in seen_lot_ids:
                                lot["confidence_factor"] = 0.6
                                all_lots.append(lot)
                                seen_lot_ids.add(lot["id"])
                                project_added += 1
            except Exception as exc:
                logger.error("SOURCE A chef de projet failed for pro_id=%s: %s", pro_id, exc, exc_info=True)

        logger.info(
            "ProTagScorer _get_lots_for_pro(%s): direct_added=%d phase_added=%d project_added=%d total=%d",
            pro_id,
            direct_added,
            phase_added,
            project_added,
            len(all_lots),
        )

        return all_lots

    def trigger_on_lot_update(self, lot_id: str) -> None:
        try:
            logger.info(f"Triggered score recalculation for lot: {lot_id}")
            # Fetch lot minimal fields nécessaires pour retrouver les responsables indirects
            response = (
                self.sb.table("lots")
                .select("id,phase_id,responsible_user_id,status")
                .eq("id", lot_id)
                .single()
                .execute()
            )
            lot_data = getattr(response, "data", None)
            lot = lot_data[0] if isinstance(lot_data, list) and lot_data else lot_data
            if not lot:
                logger.warning("Lot %s not found.", lot_id)
                return

            status = lot.get("status")
            if status not in ("en_cours", "termine", "valide"):
                logger.info("Lot ignored (status not relevant).")
                return

            pro_ids_to_update: Set[str] = set()

            # Direct
            if lot.get("responsible_user_id"):
                pro_ids_to_update.add(lot["responsible_user_id"])

            # Phase manager + Chef de projet via phase->project
            if lot.get("phase_id"):
                phase = (
                    self.sb.table("phases")
                    .select("id,project_id,phase_manager_id")
                    .eq("id", lot["phase_id"])
                    .single()
                    .execute()
                    .data
                )
                phase = phase[0] if isinstance(phase, list) and phase else phase
                if phase:
                    if phase.get("phase_manager_id"):
                        pro_ids_to_update.add(phase["phase_manager_id"])

                    if phase.get("project_id"):
                        project = (
                            self.sb.table("projects")
                            .select("project_manager_id,created_by")
                            .eq("id", phase["project_id"])
                            .single()
                            .execute()
                            .data
                        )
                        project = project[0] if isinstance(project, list) and project else project
                        if project:
                            if project.get("project_manager_id"):
                                pro_ids_to_update.add(project["project_manager_id"])
                            if project.get("created_by"):
                                pro_ids_to_update.add(project["created_by"])

            # Recalculer pour chacun (uniquement user_type == "pro")
            for pro_id in pro_ids_to_update:
                profile = (
                    self.sb.table("profiles")
                    .select("user_type")
                    .eq("id", pro_id)
                    .single()
                    .execute()
                    .data
                )
                profile = profile[0] if isinstance(profile, list) and profile else profile
                if profile and profile.get("user_type") == "pro":
                    self.compute_and_upsert_scores(pro_id=pro_id)
        except Exception as e:
            logger.error(f"Error in trigger_on_lot_update for {lot_id}: {e}", exc_info=True)

    def compute_and_upsert_scores(self, pro_id: Optional[str] = None):
        try:
            if not self.sb:
                return 0

                pro_ids_to_process: List[str]
            if pro_id:
                pro_ids_to_process = [pro_id]
            else:
                # Recalcule pour les pros potentiellement impactés par des lots "pertinents"
                relevant_statuses = ["termine", "valide", "en_cours"]

                direct_rows = (
                    self.sb.table("lots")
                    .select("responsible_user_id")
                    .in_("status", relevant_statuses)
                    .execute()
                    .data
                    or []
                )
                direct_pro_ids = {r.get("responsible_user_id") for r in direct_rows if r and r.get("responsible_user_id")}

                phase_rows = (
                    self.sb.table("lots")
                    .select("phase_id")
                    .in_("status", relevant_statuses)
                    .execute()
                    .data
                    or []
                )
                phase_ids = {r.get("phase_id") for r in phase_rows if r and r.get("phase_id")}

                phase_managers = set()
                project_ids = set()
                if phase_ids:
                    phases = (
                        self.sb.table("phases")
                        .select("id,project_id,phase_manager_id")
                        .in_("id", list(phase_ids))
                        .execute()
                        .data
                        or []
                    )
                    for p in phases:
                        if p and p.get("phase_manager_id"):
                            phase_managers.add(p["phase_manager_id"])
                        if p and p.get("project_id"):
                            project_ids.add(p["project_id"])

                projects = []
                if project_ids:
                    projects = (
                        self.sb.table("projects")
                        .select("project_manager_id,created_by")
                        .in_("id", list(project_ids))
                        .execute()
                        .data
                        or []
                    )
                project_pro_ids = set()
                for p in projects:
                    if p and p.get("project_manager_id"):
                        project_pro_ids.add(p["project_manager_id"])
                    if p and p.get("created_by"):
                        project_pro_ids.add(p["created_by"])

                member_pro_ids = set()
                if project_ids:
                    member_rows = (
                        self.sb.table("project_members")
                        .select("user_id")
                        .in_("project_id", list(project_ids))
                        .in_("status", ["accepted", "active"])
                        .in_(
                            "role",
                            [
                                "owner",
                                "collaborator",
                                "collaborateur",
                                "pro",
                                "professionnel",
                                "chef de projet",
                                "chef_de_projet",
                                "project manager",
                                "project_manager",
                            ],
                        )
                        .execute()
                        .data
                        or []
                    )
                    member_pro_ids = {r.get("user_id") for r in member_rows if r and r.get("user_id")}

                pro_ids_to_process = list(direct_pro_ids | phase_managers | project_pro_ids | member_pro_ids)

            upserts: List[dict] = []

            for r_id in pro_ids_to_process:
                lots_with_factor = self._get_lots_for_pro(r_id)

                # tag -> list of lots
                tags_dict: Dict[str, List[dict]] = {}
                for lot in lots_with_factor:
                    tag = self._normalize_tag(lot.get("lot_type"), lot.get("name"))
                    if not tag:
                        continue
                    tags_dict.setdefault(tag, []).append(lot)

                for tag, tag_lots in tags_dict.items():
                    evidence_count = len(tag_lots)
                    if evidence_count == 0:
                        continue

                    # Base score (nombre d'indices)
                    base_score = min(evidence_count / 10.0, 1.0)

                    # Completion bonus (lots terminés/validés)
                    completed_lots = sum(1 for l in tag_lots if l.get("status") in ("termine", "valide"))
                    completed_ratio = completed_lots / max(evidence_count, 1)
                    completion_bonus = completed_ratio * 0.15

                    # Delay penalty (pénalité sur le délai moyen)
                    total_delay = sum((l.get("delay_days") or 0) for l in tag_lots)
                    avg_delay = total_delay / max(evidence_count, 1)
                    delay_penalty = min(avg_delay / 30.0, 0.10)

                    # Bons bonus non disponibles dans ce modèle => placeholders
                    specialty_bonus = 0.0
                    devis_bonus = 0.0

                    confidence = base_score + completion_bonus + specialty_bonus + devis_bonus - delay_penalty

                    avg_confidence_factor = (
                        sum(l.get("confidence_factor", 1.0) for l in tag_lots) / max(len(tag_lots), 1)
                    )
                    confidence = confidence * avg_confidence_factor
                    confidence = max(0.10, min(1.0, round(confidence, 4)))

                    upserts.append(
                        {
                            "pro_id": r_id,
                            "tag": tag,
                            "confidence": confidence,
                            "evidence_count": evidence_count,
                            "source": "computed",
                            "last_seen_at": datetime.utcnow().isoformat(),
                            "updated_at": datetime.utcnow().isoformat(),
                        }
                    )

            if upserts:
                # Upsert to database (idempotent sur pro_id + tag)
                for batch in [upserts[i : i + 500] for i in range(0, len(upserts), 500)]:
                    self.sb.table("pro_tag_scores").upsert(batch, on_conflict="pro_id,tag").execute()

                logger.info("Updated %d pro logic tags.", len(upserts))
            else:
                logger.info("No tags updated.")

            return len(upserts)
        except Exception as e:
            logger.error(f"Error computing pro tag scores: {e}", exc_info=True)
            return 0
