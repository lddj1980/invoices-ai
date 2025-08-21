import sys
from pathlib import Path

import pytest

# Ensure the project root is on the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.agents import (
    SALE_CONDITIONS,
    SALES_CONDITION_PROMPT,
    normalize_sales_condition,
    exporter_is_manufacturer,
    select_relevant_chunks,
    identify_field,
)


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


def test_exporter_is_manufacturer_detects_phrase():
    text = "Some info THE EXPORTER IS THE SAME AS THE MANUFACTURER end."
    assert exporter_is_manufacturer(text) is True


def test_exporter_is_manufacturer_when_phrase_absent():
    assert exporter_is_manufacturer("No relation mentioned.") is False


def test_select_relevant_chunks_returns_best_match():
    chunks = [
        "exporter manufacturer",
        "other data",
        "manufacturer exporter info",
    ]
    result = select_relevant_chunks("exporter", chunks, top_k=1)
    assert result == ["exporter manufacturer"]


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
