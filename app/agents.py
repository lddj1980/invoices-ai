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
    list of str
        The ``top_k`` chunks ordered from most to least relevant.
    """

    from difflib import SequenceMatcher

    scored = [
        (SequenceMatcher(None, query.lower(), chunk.lower()).ratio(), chunk)
        for chunk in chunks
    ]
    scored.sort(reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


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

    relevant = select_relevant_chunks(field_name, knowledge_base)
    context = "\n---\n".join(relevant)
    prompt = (
        f"{rules}\n\nContext:\n{context}\n\nDocument:\n{document}\n"
        f"Extract the value for '{field_name}'."
    )

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
