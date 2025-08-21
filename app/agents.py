SALE_CONDITIONS = {
    "CFR": "Cost and Freight",
    "CIF": "Cost, Insurance and Freight",
    "CIP": "Carriage and Insurance Paid To",
    "CPT": "Carriage Paid To",
    "C_F": "Custo e Frete",
    "C_I": "Custo e Seguro",
    "DAP": "Delivered At Place",
    "DAT": "Delivered At Terminal",
    "EXW": "Ex Works",
    "FAS": "Free Alongside Ship",
    "FCA": "Free Carrier",
    "FOB": "Free On Board",
    "OCV": "Outra Condição de Venda (ver README.md#condições-de-venda-suportadas)",
}

# Prompt used to capture the sales condition from a user or external system.
SALES_CONDITION_PROMPT = (
    "Informe a condição de venda (" + ", ".join(SALE_CONDITIONS.keys()) + "). "
    "Consulte README.md#condições-de-venda-suportadas:"
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
    raise ValueError(
        "Condição de venda desconhecida. Consulte README.md#condições-de-venda-suportadas"
    )
