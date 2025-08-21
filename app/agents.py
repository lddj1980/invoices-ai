SALE_CONDITIONS = {
    "CFR": "...",
    "CIF": "...",
    "CIP": "...",
    "CPT": "...",
    "C_F": "...",
    "C_I": "...",
    "DAP": "...",
    "DAT": "...",
    "EXW": "...",
    "FAS": "...",
    "FCA": "...",
    "FOB": "...",
    "OCV": "...",
}

# Prompt used to capture the sales condition from a user or external system.
SALES_CONDITION_PROMPT = (
    "Informe a condição de venda (" + ", ".join(SALE_CONDITIONS.keys()) + "):"
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
    if code in SALE_CONDITIONS:
        return code
    raise ValueError("Condição de venda desconhecida")


def exporter_is_manufacturer(text: str) -> bool:
    """Check whether a document states that exporter and manufacturer are the same.

    The check is performed by searching for the phrase
    ``"the exporter is the same as the manufacturer"`` in a case-insensitive
    manner. Extraneous whitespace is ignored so that the match is resilient to
    formatting differences in the source text.

    Parameters
    ----------
    text: str
        Text extracted from an invoice or related document.

    Returns
    -------
    bool
        ``True`` if the phrase is present, otherwise ``False``.
    """

    normalized_text = " ".join(text.lower().split())
    phrase = "the exporter is the same as the manufacturer"
    return phrase in normalized_text
