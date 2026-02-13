"""Tests unitaires minimaux pour l'agent IA (sans serveur ni base de données)."""

import json
import sys
from pathlib import Path

import pytest

# Remonter au dossier parent pour que les imports fonctionnent
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==== _maybe_parse_json (runtime.py) ====

from agent.runtime import _maybe_parse_json


class TestMaybeParseJson:
    """Tests pour _maybe_parse_json."""

    def test_valid_json(self):
        result = _maybe_parse_json('{"reply": "ok", "todo": []}')
        assert isinstance(result, dict)
        assert result["reply"] == "ok"

    def test_invalid_text_returns_none(self):
        result = _maybe_parse_json("ceci n'est pas du json")
        assert result is None

    def test_extract_json_from_mixed_text(self):
        text = 'Voici ma réponse: {"reply": "bonjour", "todo": ["action1"]} fin du texte'
        result = _maybe_parse_json(text)
        assert isinstance(result, dict)
        assert result["reply"] == "bonjour"
        assert result["todo"] == ["action1"]

    def test_empty_string_returns_none(self):
        result = _maybe_parse_json("")
        assert result is None

    def test_nested_json(self):
        data = {"reply": "test", "document": {"customer": {"name": "Dupont"}}}
        result = _maybe_parse_json(json.dumps(data))
        assert isinstance(result, dict)
        assert result["document"]["customer"]["name"] == "Dupont"

    def test_json_with_escaped_chars(self):
        text = '{"reply": "c\'est un test avec \\"guillemets\\"", "todo": []}'
        result = _maybe_parse_json(text)
        assert isinstance(result, dict)
        assert "test" in result["reply"]


# ==== _format_ai_reply (api.py) ====

from agent.api import _format_ai_reply


class TestFormatAiReply:
    """Tests pour _format_ai_reply."""

    def test_string_input(self):
        result = _format_ai_reply("bonjour")
        assert isinstance(result, str)
        assert "bonjour" in result

    def test_dict_with_reply(self):
        result = _format_ai_reply({"reply": "voici ma réponse", "todo": []})
        assert isinstance(result, str)
        assert "voici ma réponse" in result

    def test_none_input(self):
        result = _format_ai_reply(None)
        assert isinstance(result, str)

    def test_dict_without_reply(self):
        result = _format_ai_reply({"document": {}, "corrections": []})
        assert isinstance(result, str)


# ==== Prompts files exist ====

PROMPTS_DIR = ROOT / "prompts"


class TestPromptsExist:
    """Vérifie que les fichiers de prompts essentiels existent."""

    @pytest.mark.parametrize(
        "filename",
        [
            "chat_prompt.txt",
            "analysis_prompt.txt",
            "validate_prompt.txt",
            "prepare_devis_prompt.txt",
        ],
    )
    def test_prompt_file_exists(self, filename: str):
        path = PROMPTS_DIR / filename
        assert path.exists(), f"Fichier prompt manquant: {path}"


# ==== Config sanity ====


class TestConfig:
    """Vérifie les valeurs par défaut critiques."""

    def test_default_model_is_valid(self):
        from agent.config import DEFAULT_MODEL
        assert DEFAULT_MODEL is not None
        assert "gpt-5-mini" not in DEFAULT_MODEL, "gpt-5-mini n'est pas un modèle OpenAI valide"
        assert len(DEFAULT_MODEL) > 0

    def test_env_example_exists(self):
        env_example = ROOT / ".env.example"
        assert env_example.exists(), ".env.example manquant"
