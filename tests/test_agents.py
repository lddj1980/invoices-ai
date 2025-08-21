import sys
from pathlib import Path

import pytest

# Ensure the project root is on the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.agents import SALE_CONDITIONS, SALES_CONDITION_PROMPT, normalize_sales_condition


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
