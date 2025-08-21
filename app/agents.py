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
    raise ValueError("Condição de venda desconhecida")


def exporter_is_manufacturer(text: str) -> bool:
    """Return ``True`` if the text states exporter and manufacturer are the same.

    The detection is intentionally conservative and relies on a small set of
    regular expressions that cover common ways of expressing that the exporter
    and manufacturer are a single entity.  Both English and Portuguese phrases
    are supported.

    Parameters
    ----------
    text:
        Text extracted from an invoice or related document.

    Returns
    -------
    bool
        ``True`` if a known phrase is detected, otherwise ``False``.
    """

    import re

    normalized_text = " ".join(text.lower().split())

    patterns = [
        r"the\s+exporter\s+is\s+the\s+same\s+as\s+the\s+manufacturer",
        r"the\s+exporter\s+is\s+also\s+the\s+manufacturer",
        r"the\s+exporter\s+and\s+the\s+manufacturer\s+are\s+the\s+same",
        r"the\s+exporter\s+is\s+the\s+manufacturer",
        r"o\s+exportador\s+é\s+o\s+mesmo\s+que\s+o\s+fabricante",
        r"o\s+exportador\s+é\s+tamb[eé]m\s+o\s+fabricante",
        r"o\s+exportador\s+e\s+o\s+fabricante\s+são\s+o\s+mesmo",
        r"o\s+exportador\s+é\s+o\s+fabricante",
    ]

    return any(re.search(pattern, normalized_text) for pattern in patterns)


def exportador_fabricante(text: str) -> bool:
    """Convenience wrapper to keep a Portuguese friendly API name."""

    return exporter_is_manufacturer(text)


def possible_company_names(text: str):
    """Return possible company names not preceded by ``To``.

    The function applies a simple heuristic over each line of ``text``
    looking for common company suffixes (e.g. ``Ltda``, ``SA``, ``Inc``).
    Lines that start with ``To`` are ignored to avoid capturing the
    recipient of the document.  Matching is case-insensitive and the
    detected names are returned in the order they appear.

    Parameters
    ----------
    text:
        Raw text extracted from an invoice or related document.

    Returns
    -------
    list of str
        List of company-like names found in ``text``.
    """

    import re

    suffixes = r"(?:S\.A\.?|SA|LTDA|Ltda|LTD|Ltd|Inc\.?|INC|LLC|Corp\.?|Corporation|Company|Co\.?|GmbH)"
    pattern = re.compile(rf"\b([A-Z][A-Za-z0-9&.,'\s]+?{suffixes})\b")

    names = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("to"):
            continue
        match = pattern.search(stripped)
        if match:
            names.append(match.group(1).strip())
    return names


def select_relevant_chunks(query: str, chunks, top_k: int = 3):
    """Return the most relevant chunks using a simple lexical similarity.

    This function implements the retrieval step of RAG (Retrieval-Augmented
    Generation) by selecting ``top_k`` chunks that best match the query using
    :class:`difflib.SequenceMatcher`.

    Parameters
    ----------
    query:
        Search query representing what information is needed.
    chunks:
        Iterable of text segments to search.
    top_k:
        Number of chunks to return.

    Returns
    -------
    list of tuple
        Tuples of ``(score, chunk)`` ordered from most to least relevant.
    """

    from difflib import SequenceMatcher

    scored = [
        (SequenceMatcher(None, query.lower(), chunk.lower()).ratio(), chunk)
        for chunk in chunks
    ]
    scored.sort(reverse=True)
    return scored[:top_k]


def identify_field(field_name: str, document: str, rules: str, knowledge_base):
    """Use OpenAI's LLM to identify a field in ``document``.

    The function retrieves ``OPENAI_API_KEY`` and ``OPENAI_API_MODEL`` from the
    environment and uses a simple RAG approach: ``knowledge_base`` is searched
    for the chunks most relevant to ``field_name`` and provided to the model as
    context.

    Parameters
    ----------
    field_name:
        Name of the field to extract.
    document:
        Text of the document where the field should be found.
    rules:
        Instructions describing how the field should be identified.
    knowledge_base:
        Iterable of text chunks used for retrieval.

    Returns
    -------
    str
        The model's response for the requested field.
    """

    import os

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_API_MODEL")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    if not model:
        raise EnvironmentError("OPENAI_API_MODEL is not set")

    # Import OpenAI lazily so tests can run without the package installed and
    # instantiate a client using the new 1.x interface.
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    query = f"{field_name} {rules}".strip()
    relevant = select_relevant_chunks(query, knowledge_base)
    if not relevant or relevant[0][0] < 0.3:
        context = document
    else:
        context = "\n---\n".join(chunk for _, chunk in relevant)

    prompt = (
        f"{rules}\n\nContext:\n{context}\n\nDocument:\n{document}\n"
        f"Extract the value for '{field_name}'."
    )

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def identify_fields(
    document: str, field_rules: dict, knowledge_base
):
    """Identify multiple fields in ``document`` using separate RAG calls.

    For each ``field_name`` and its ``rules`` in ``field_rules`` a dedicated
    :func:`identify_field` invocation is performed so that context retrieval and
    LLM generation happen independently.  The collected results are returned in a
    dictionary mapping field names to the extracted values.

    Parameters
    ----------
    document:
        Full text of the source document.
    field_rules:
        Mapping of field names to the extraction rules for each field.
    knowledge_base:
        Iterable of text chunks used for retrieval.

    Returns
    -------
    dict
        Dictionary with one entry per ``field_rules`` item, containing the
        extracted values.
    """

    results = {}
    for field_name, rules in field_rules.items():
        results[field_name] = identify_field(
            field_name, document, rules, knowledge_base
        )
    return results


def identify_products(document: str, product_rules: dict, knowledge_base):
    """Identify a list of products in ``document`` using RAG calls.

    Each entry in ``product_rules`` describes how to extract a particular
    attribute (e.g. description or quantity).  The associated LLM responses are
    expected to contain one value per line, allowing this function to assemble
    the individual attributes into a list of product dictionaries.

    Parameters
    ----------
    document:
        Full text of the source document.
    product_rules:
        Mapping of product field names to extraction rules.  Each rule should
        instruct the model to return one value per line.
    knowledge_base:
        Iterable of text chunks used for retrieval.

    Returns
    -------
    list of dict
        A list where each element is a dictionary representing a product and
        containing the fields specified in ``product_rules``.
    """

    extracted = {}
    max_len = 0
    for field_name, rules in product_rules.items():
        value = identify_field(field_name, document, rules, knowledge_base)
        items = [item.strip() for item in value.splitlines() if item.strip()]
        extracted[field_name] = items
        if len(items) > max_len:
            max_len = len(items)

    products = []
    for i in range(max_len):
        product = {}
        for field_name, values in extracted.items():
            product[field_name] = values[i] if i < len(values) else None
        products.append(product)

    return products
