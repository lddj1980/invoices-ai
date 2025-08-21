from enum import Enum


class SalesCondition(str, Enum):
    """Enum representing supported sales conditions."""
    CFR = "CFR"
    CIF = "CIF"
    CIP = "CIP"
    CPT = "CPT"
    C_F = "C_F"
    C_I = "C_I"
    DAP = "DAP"
    DAT = "DAT"
    EXW = "EXW"
    FAS = "FAS"
    FCA = "FCA"
    FOB = "FOB"
    OCV = "OCV"


# Prompt used to capture the sales condition from a user or external system.
# This replaces the previous hard coded "condição de venda como FOB" string
# with a generic message that lists every supported option.
SALES_CONDITION_PROMPT = (
    "Informe a condição de venda (CFR, CIF, CIP, CPT, C_F, C_I, DAP, DAT, EXW, "
    "FAS, FCA, FOB, OCV):"
)


def normalize_sales_condition(value: str) -> str:
    """Validate and normalize a sales condition code.

    Parameters
    ----------
    value: str
        Input sales condition code which may be in any case.

    Returns
    -------
    str
        Upper-case sales condition code if valid.

    Raises
    ------
    ValueError
        If ``value`` is not a supported sales condition.
    """
    code = value.strip().upper()
    try:
        return SalesCondition(code).value
    except ValueError as exc:  # pragma: no cover - error message is enough
        raise ValueError(f"Condição de venda inválida: {value}") from exc
