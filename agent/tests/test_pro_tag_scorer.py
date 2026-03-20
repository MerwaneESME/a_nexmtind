import pytest
from unittest.mock import MagicMock

# Assuming standard pytest structure. Need to mock supabase.
from agent.services.pro_tag_scorer import ProTagScorer

def test_tag_mapping_normalization():
    scorer = ProTagScorer()
    assert scorer._normalize_tag("électricité", None) == "electricite"
    assert scorer._normalize_tag("maconnerie", None) == "maconnerie"
    assert scorer._normalize_tag("", "Gros oeuvre complet") == "gros_oeuvre"
    assert scorer._normalize_tag("unknown", "Installation de chauffage") == "chauffage"

def test_confidence_calculation():
    # Mocking supabase client
    scorer = ProTagScorer()
    scorer.supabase = MagicMock()
    scorer.sb = scorer.supabase
    
    # 5 lots terminés sans retard (responsabilité directe => confidence_factor=1.0)
    mock_lots = [
        {"lot_type": "plomberie", "status": "termine", "delay_days": 0, "confidence_factor": 1.0},
        {"lot_type": "plomberie", "status": "termine", "delay_days": 0, "confidence_factor": 1.0},
        {"lot_type": "plomberie", "status": "termine", "delay_days": 0, "confidence_factor": 1.0},
        {"lot_type": "plomberie", "status": "termine", "delay_days": 0, "confidence_factor": 1.0},
        {"lot_type": "plomberie", "status": "termine", "delay_days": 0, "confidence_factor": 1.0},
    ]
    scorer._get_lots_for_pro = MagicMock(return_value=mock_lots)
    
    # We mock upsert to capture the arguments
    upsert_mock = MagicMock()
    scorer.supabase.table.return_value.upsert.return_value.execute = upsert_mock
    
    scorer.compute_and_upsert_scores("pro1")
    
    # Get the arguments passed to upsert
    call_args = scorer.supabase.table.return_value.upsert.call_args
    assert call_args is not None
    batch = call_args[0][0]
    
    assert len(batch) == 1
    score = batch[0]
    assert score["tag"] == "plomberie"
    assert score["pro_id"] == "pro1"
    assert score["evidence_count"] == 5

    # On vérifie surtout que la confidence est dans le range attendu.
    assert isinstance(score["confidence"], (int, float))
    assert 0.10 <= score["confidence"] <= 1.0

def test_trigger_skips_null_responsible():
    scorer = ProTagScorer()
    scorer.supabase = MagicMock()
    scorer.sb = scorer.supabase
    
    mock_lot = {
        "id": "lot1",
        "responsible_user_id": None,
        "phase_id": None,
        "status": "termine"
    }
    
    mock_response = MagicMock()
    mock_response.data = mock_lot
   
    mock_query = scorer.supabase.table.return_value
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.single.return_value = mock_query
    mock_query.execute.return_value = mock_response
    
    # Mock compute_and_upsert_scores to ensure it's NOT called
    scorer.compute_and_upsert_scores = MagicMock()
    
    scorer.trigger_on_lot_update("lot1")
    
    scorer.compute_and_upsert_scores.assert_not_called()


def test_chef_de_projet_gets_reduced_confidence():
    """Un chef de projet obtient confidence_factor=0.6 sur les lots du projet"""
    scorer = ProTagScorer()
    scorer.supabase = MagicMock()
    scorer.sb = scorer.supabase

    scorer.supabase.table.return_value.upsert.return_value.execute = MagicMock()

    base_lots = [{"lot_type": "plomberie", "status": "termine", "delay_days": 0} for _ in range(5)]

    scorer._get_lots_for_pro = MagicMock(
        return_value=[{**l, "confidence_factor": 1.0} for l in base_lots]
    )
    scorer.compute_and_upsert_scores("pro1")
    direct_conf = scorer.supabase.table.return_value.upsert.call_args[0][0][0]["confidence"]

    scorer._get_lots_for_pro = MagicMock(
        return_value=[{**l, "confidence_factor": 0.6} for l in base_lots]
    )
    scorer.compute_and_upsert_scores("pro1")
    reduced_conf = scorer.supabase.table.return_value.upsert.call_args[0][0][0]["confidence"]

    assert reduced_conf < direct_conf
    expected = round(direct_conf * 0.6, 4)
    assert abs(reduced_conf - expected) <= 0.02


def test_phase_manager_gets_medium_confidence():
    """Un responsable de phase obtient confidence_factor=0.8"""
    scorer = ProTagScorer()
    scorer.supabase = MagicMock()
    scorer.sb = scorer.supabase

    scorer.supabase.table.return_value.upsert.return_value.execute = MagicMock()

    base_lots = [{"lot_type": "plomberie", "status": "termine", "delay_days": 0} for _ in range(5)]

    scorer._get_lots_for_pro = MagicMock(
        return_value=[{**l, "confidence_factor": 1.0} for l in base_lots]
    )
    scorer.compute_and_upsert_scores("pro1")
    direct_conf = scorer.supabase.table.return_value.upsert.call_args[0][0][0]["confidence"]

    scorer._get_lots_for_pro = MagicMock(
        return_value=[{**l, "confidence_factor": 0.8} for l in base_lots]
    )
    scorer.compute_and_upsert_scores("pro1")
    reduced_conf = scorer.supabase.table.return_value.upsert.call_args[0][0][0]["confidence"]

    assert reduced_conf < direct_conf
    expected = round(direct_conf * 0.8, 4)
    assert abs(reduced_conf - expected) <= 0.02
