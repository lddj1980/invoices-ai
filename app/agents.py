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

    The function retrieves ``OPENAI_API_KEY`` from the environment and uses a
    simple RAG approach: ``knowledge_base`` is searched for the chunks most
    relevant to ``field_name`` and provided to the model as context.

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
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    # Import OpenAI lazily so tests can run without the package installed.
    import openai

    openai.api_key = api_key

    relevant = select_relevant_chunks(field_name, knowledge_base)
    context = "\n---\n".join(relevant)
    prompt = (
        f"{rules}\n\nContext:\n{context}\n\nDocument:\n{document}\n"
        f"Extract the value for '{field_name}'."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()
