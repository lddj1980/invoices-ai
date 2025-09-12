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
    extract_product_headers,
    extract_products,
    identify_products,
)
from app.main import FIELD_RULES
from app.prompts import INVOICE_PARSER_STEP1_PROMPT


def test_prompt_lists_all_conditions():
    for condition in SALE_CONDITIONS:
        assert condition in SALES_CONDITION_PROMPT


def test_invoice_prompt_enforces_exporter_manufacturer_rules():
    assert 'EXPORTADOR É O FABRICANTE?"=true' in INVOICE_PARSER_STEP1_PROMPT
    assert '"FABRICANTE É CONHECIDO?"=false' in INVOICE_PARSER_STEP1_PROMPT
    assert 'Se "EXPORTADOR É O FABRICANTE?" = false, procure explicitamente na invoice o "NOME DO FABRICANTE" e o "PAIS DO FABRICANTE"' in INVOICE_PARSER_STEP1_PROMPT
def test_invoice_prompt_defines_partnumber():
    assert '"PARTNUMBER" (string) → "descrição do produto+size"' in INVOICE_PARSER_STEP1_PROMPT

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


def test_identify_field_missing_openai_package(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_MODEL", "gpt-test")
    monkeypatch.setitem(sys.modules, "openai", None)
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

    dummy_openai = SimpleNamespace(OpenAI=lambda api_key, **kwargs: DummyClient())

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


def test_extract_product_headers_maps_aliases(monkeypatch):
    def fake_identify_field(field_name, document, rules, kb):
        assert field_name == "product_headers"
        return "NO., DBSCRIPTION, QTY(PC), G.W.(KGs)"

    monkeypatch.setattr("app.agents.identify_field", fake_identify_field)

    result = extract_product_headers("doc", ["kb"])
    assert result["raw_headers_detected"] == [
        "NO.",
        "DBSCRIPTION",
        "QTY(PC)",
        "G.W.(KGs)",
    ]
    assert result["mapping"]["NO."] == "no"
    assert result["mapping"]["DBSCRIPTION"] == "description"
    assert result["mapping"]["QTY(PC)"] == "qty_pc"
    assert result["mapping"]["G.W.(KGs)"] == "gross_weight_kg"


def test_extract_products_parses_details_and_totals(monkeypatch):
    headers = {
        "raw_headers_detected": ["NO.", "DESC", "QTY(PC)", "N.W.(KGs)", "G.W.(KGs)"],
        "normalized_headers": [
            "no",
            "description",
            "qty_pc",
            "net_weight_kg",
            "gross_weight_kg",
        ],
        "mapping": {
            "NO.": "no",
            "DESC": "description",
            "QTY(PC)": "qty_pc",
            "N.W.(KGs)": "net_weight_kg",
            "G.W.(KGs)": "gross_weight_kg",
        },
    }

    def fake_identify_field(field_name, document, rules, kb):
        import json

        assert field_name == "products"
        return json.dumps(
            {
                "products": [
                    {
                        "no": "1",
                        "description": "A",
                        "details": [
                            {"size_spec": "M", "qty_pc": "10", "net_weight_kg": "5,0"},
                            {"size_spec": "L", "qty_pc": "20", "net_weight_kg": "10"},
                        ],
                    }
                ],
                "totals_detected": {"total_net_weight_kg": "15", "pallets_count": "3"},
            }
        )

    monkeypatch.setattr("app.agents.identify_field", fake_identify_field)

    result = extract_products("doc", headers, ["kb"])
    assert result["products"][0]["no"] == "1"
    assert result["products"][0]["details"][0]["seq_no"] == 1
    assert result["products"][0]["details"][1]["seq_no"] == 2
    assert result["products"][0]["details"][0]["qty_pc"] == 10
    assert result["products"][0]["details"][0]["net_weight_kg"] == 5.0
    assert result["totals_detected"]["total_net_weight_kg"] == 15
    assert result["totals_detected"]["pallets_count"] == 3


def test_identify_products_combines(monkeypatch):
    calls = []
    headers = {"raw_headers_detected": [], "normalized_headers": [], "mapping": {}}

    def fake_extract_product_headers(document, kb):
        calls.append(("headers", document, kb))
        return headers

    def fake_extract_products(document, hdrs, kb):
        calls.append(("products", document, hdrs, kb))
        return {"products": [], "totals_detected": {}}

    monkeypatch.setattr(
        "app.agents.extract_product_headers", fake_extract_product_headers
    )
    monkeypatch.setattr("app.agents.extract_products", fake_extract_products)

    result = identify_products("doc", ["kb"])
    assert result == {"products": [], "totals_detected": {}}
    assert calls == [
        ("headers", "doc", ["kb"]),
        ("products", "doc", headers, ["kb"]),
    ]
