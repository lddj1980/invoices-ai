import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure the project root is on the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.agents import (
    SALE_CONDITIONS,
    SALES_CONDITION_PROMPT,
    normalize_sales_condition,
    exportador_fabricante,
    possible_company_names,
    select_relevant_chunks,
    identify_field,
    identify_fields,
    identify_products,
)
from app.main import FIELD_RULES


def test_prompt_lists_all_conditions():
    for condition in SALE_CONDITIONS:
        assert condition in SALES_CONDITION_PROMPT


def test_normalize_sales_condition_fob():
    assert normalize_sales_condition("FOB") == "FOB"


def test_normalize_sales_condition_cfr_case_insensitive_and_spaces():
    assert normalize_sales_condition("  cfr  ") == "CFR"


def test_invalid_condition_raises_error():
    with pytest.raises(ValueError):
        normalize_sales_condition("XYZ")


def test_exportador_fabricante_detects_phrase():
    text = "Some info THE EXPORTER IS THE SAME AS THE MANUFACTURER end."
    assert exportador_fabricante(text) is True


def test_exportador_fabricante_detects_variation():
    text = "Details: The exporter is also the manufacturer of the goods."
    assert exportador_fabricante(text) is True


def test_exportador_fabricante_when_phrase_absent():
    assert exportador_fabricante("No relation mentioned.") is False


def test_possible_company_names_ignores_to_lines():
    text = "ACME LTDA\nTo: Importer SA\nExporter LLC"
    assert possible_company_names(text) == ["ACME LTDA", "Exporter LLC"]


def test_select_relevant_chunks_returns_best_match():
    chunks = [
        "exporter manufacturer",
        "other data",
        "manufacturer exporter info",
    ]
    result = select_relevant_chunks("exporter", chunks, top_k=1)
    assert result[0][1] == "exporter manufacturer"
    assert 0 <= result[0][0] <= 1


def test_identify_field_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_MODEL", "gpt-test")
    with pytest.raises(EnvironmentError):
        identify_field("field", "doc", "rules", ["chunk"])


def test_identify_field_requires_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_API_MODEL", raising=False)
    with pytest.raises(EnvironmentError):
        identify_field("field", "doc", "rules", ["chunk"])


def test_field_rules_has_exportador_fabricante():
    assert "exportador_fabricante" in FIELD_RULES


def test_field_rules_has_nome_do_exportador():
    assert "nome_do_exportador" in FIELD_RULES


def test_identify_field_uses_rules_and_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_MODEL", "gpt-test")

    captured = {}

    def fake_select(query, chunks, top_k=3):
        captured["query"] = query
        return [(0.1, "irrelevant")]

    class DummyCompletions:
        def create(self, model, messages):
            captured["prompt"] = messages[0]["content"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="result"))
                ]
            )

    class DummyClient:
        def __init__(self, *a, **k):
            self.chat = SimpleNamespace(completions=DummyCompletions())

    dummy_openai = SimpleNamespace(OpenAI=lambda api_key: DummyClient())

    monkeypatch.setattr("app.agents.select_relevant_chunks", fake_select)
    monkeypatch.setitem(sys.modules, "openai", dummy_openai)

    identify_field("field", "full document", "rules", ["chunk"])

    assert captured["query"] == "field rules"
    context_section = captured["prompt"].split("Context:\n")[1].split("\n\nDocument:")[0]
    assert context_section == "full document"


def test_identify_fields_invokes_identify_field(monkeypatch):
    calls = []

    def fake_identify_field(field_name, document, rules, knowledge_base):
        calls.append((field_name, document, rules, knowledge_base))
        return f"value-{field_name}"

    monkeypatch.setattr("app.agents.identify_field", fake_identify_field)

    result = identify_fields(
        "doc",
        {"a": "rule A", "b": "rule B"},
        ["kb"],
    )
    assert result == {"a": "value-a", "b": "value-b"}
    assert calls == [
        ("a", "doc", "rule A", ["kb"]),
        ("b", "doc", "rule B", ["kb"]),
    ]


def test_identify_products_zips_fields(monkeypatch):
    calls = {}

    def fake_identify_field(field_name, document, rules, knowledge_base):
        calls[field_name] = (document, rules, knowledge_base)
        if field_name == "descricao":
            return "A\nB"
        if field_name == "quantidade":
            return "1\n2"
        if field_name == "tamanho":
            return "S\nM"

    monkeypatch.setattr("app.agents.identify_field", fake_identify_field)

    rules = {"descricao": "d", "quantidade": "q", "tamanho": "t"}
    result = identify_products("doc", rules, ["kb"])

    assert result == [
        {"descricao": "A", "quantidade": "1", "tamanho": "S"},
        {"descricao": "B", "quantidade": "2", "tamanho": "M"},
    ]
    for field_name in rules:
        assert calls[field_name] == ("doc", rules[field_name], ["kb"])
