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


# Prompts used by the product extraction agents. Prompt 1 identifies and
# normalizes table headers, while Prompt 2 extracts the structured products and
# totals using those headers.
PRODUCT_HEADERS_PROMPT = (
    "Você é um extrator de metadados de tabelas em documentos de comércio "
    "exterior (packing list).\n"
    "Dado um texto bruto (com quebras de linha e possíveis erros de OCR), "
    "identifique somente os nomes das colunas da tabela de itens, "
    "padronize-os e retorne um JSON com:\n\n"
    "raw_headers_detected: lista com todos os cabeçalhos/frases reconhecidos "
    "(na ordem de aparição, sem deduplicar).\n\n"
    "normalized_headers: lista única com mapeamento normalizado recomendado, "
    "usando exatamente estes nomes quando presentes:\n\n"
    "no, description, size_spec, qty_pc, pcs_per_ctn, ctn_per_plt, plts, "
    "net_weight_kg, gross_weight_kg, volume_cbm\n\n"
    "mapping: objeto { <raw>: <normalized> } com o mapeamento realizado.\n\n"
    "Regras duras:\n\n"
    "Não invente colunas. Se uma coluna não existir no texto, não inclua em "
    "normalized_headers.\n\n"
    "Considere variações e ruído de OCR, mapeando para os nomes acima. "
    "Exemplos de alias comuns:\n\n"
    "NO. → no\n\n"
    "DESCRIPTION, DESC, DBSCRIPTION, DESCRIÇÃO → description\n\n"
    "SIZE, SIZEQTY, SIZES → size_spec\n\n"
    "QTY., QTY, QTY(PC), QTY (PC), QTY.(PC) → qty_pc\n\n"
    "PCS/CTN, PCS / CTN, PCSCTN → pcs_per_ctn\n\n"
    "CTN/PLT, CTN / PLT, CTNPLT → ctn_per_plt\n\n"
    "PLTS, PLT, PALLETS → plts\n\n"
    "N.W. (KGS), NW, N.W., N.W.(KGs) → net_weight_kg\n\n"
    "G.W. (KGS), GW, G.W., G.W.(KGs) → gross_weight_kg\n\n"
    "Volume (CBM), CBM, cBlvt, cBIvt, cBrvr → volume_cbm\n\n"
    "Saída apenas JSON válido, sem texto extra.\n\n"
    "Entrada: {TEXTO_BRUTO}"
)

PRODUCT_ITEMS_PROMPT = (
    "Você é um extrator estruturado de packing list.\n"
    "A partir do texto bruto e do conjunto de nomes de colunas "
    "detectadas/normalizadas (do prompt anterior), extraia os itens em uma "
    "estrutura de cabeçalho + detalhes:\n\n"
    "Esquema de saída\n\n"
    "Retorne apenas um JSON com:\n\n"
    "{\n"
    "  \"products\": [\n"
    "    {\n"
    "      \"no\": \"string|null\",\n"
    "      \"description\": \"string\",\n"
    "      \"details\": [\n"
    "        {\n"
    "          \"seq_no\": 1,\n"
    "          \"size_spec\": \"string|null\",\n"
    "          \"qty_pc\": \"number|null\",\n"
    "          \"pcs_per_ctn\": \"number|null\",\n"
    "          \"ctn_per_plt\": \"number|null\",\n"
    "          \"plts\": \"number|null\",\n"
    "          \"net_weight_kg\": \"number|null\",\n"
    "          \"gross_weight_kg\": \"number|null\",\n"
    "          \"volume_cbm\": \"number|null\"\n"
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ],\n"
    "  \"totals_detected\": {\n"
    "    \"total_net_weight_kg\": \"number|null\",\n"
    "    \"total_gross_weight_kg\": \"number|null\",\n"
    "    \"total_volume_cbm\": \"number|null\",\n"
    "    \"pallets_count\": \"number|null\"\n"
    "  }\n"
    "}\n\n"
    "Regras de parsing\n\n"
    "Descrição como cabeçalho: uma nova description indica um novo product.\n\n"
    "Múltiplas sequências: após a description, podem aparecer várias linhas de "
    "detalhe para o mesmo produto (tamanhos diferentes, pesos/quantidades "
    "diferentes, etc.). Cada linha vira um objeto em details com seq_no "
    "crescente a partir de 1.\n\n"
    "no: quando houver um marcador de item como t#, 2#, 14#, NO. etc., use-o "
    "(texto puro). Se ausente, use null.\n\n"
    "size_spec: mantenha o texto integral do tamanho/especificação (ex.: M10, "
    "1\" (24–26mm), 1.1/4\"–1.3/8\" (32–36mm)).\n\n"
    "Números: converter para número (ponto decimal). Troque vírgula por ponto. "
    "Remova unidades (ex.: KGs).\n\n"
    "Campos ausentes → null.\n\n"
    "Corrija ruídos comuns de OCR quando inequívocos (ex.: cBIvt→CBM, DBSCRIPTION→DESCRIPTION, G.W→G.W.).\n\n"
    "Não deduza pesos/volumes quando não estiverem presentes na mesma linha do "
    "detalhe.\n\n"
    "Não remova informação relevante do texto da descrição.\n\n"
    "Se houver totais (ex.: TOTAL NET WEIGHT, TOTAL GROSS WEIGHT, MEASUREMENT, "
    "TOTAL: 32 PALLETS), preencha totals_detected.\n\n"
    "Validação\n\n"
    "Cada objeto em products[].details[] deve ter pelo menos um de: size_spec, "
    "qty_pc, net_weight_kg, gross_weight_kg, volume_cbm.\n\n"
    "products[].description nunca pode ser vazia.\n\n"
    "JSON válido, sem texto extra.\n\n"
    "Entrada:\n\n"
    "raw_text: {TEXTO_BRUTO}\n\n"
    "normalized_headers: {NORMALIZED_HEADERS}"
)

# Aliases used to map header names found in documents to normalized keys.
HEADER_ALIASES = {
    "no": {"no.", "no"},
    "description": {"description", "desc", "dbscription", "descrição"},
    "size_spec": {"size", "sizeqty", "sizes"},
    "qty_pc": {"qty.", "qty", "qty(pc)", "qty (pc)", "qty.(pc)"},
    "pcs_per_ctn": {"pcs/ctn", "pcs / ctn", "pcsctn"},
    "ctn_per_plt": {"ctn/plt", "ctn / plt", "ctnplt"},
    "plts": {"plts", "plt", "pallets"},
    "net_weight_kg": {"n.w. (kgs)", "nw", "n.w.", "n.w.(kgs)"},
    "gross_weight_kg": {"g.w. (kgs)", "gw", "g.w.", "g.w.(kgs)"},
    "volume_cbm": {"volume (cbm)", "cbm", "cblvt", "cbivt", "cbrvr"},
}


def _normalize_header(value: str) -> str:
    """Return a case- and accent-insensitive representation of ``value``."""

    import unicodedata

    normalized = unicodedata.normalize("NFD", value)
    without_accents = "".join(
        c for c in normalized if unicodedata.category(c) != "Mn"
    )
    return without_accents.lower().strip()


def extract_product_headers(document: str, knowledge_base):
    """Identify product table headers and map them to normalized names."""

    import re
    rules = PRODUCT_HEADERS_PROMPT.replace("{TEXTO_BRUTO}", document)
    response = identify_field("product_headers", "", rules, knowledge_base)
    headers = [h.strip() for h in re.split(r"[,\n]", response) if h.strip()]

    mapping = {}
    normalized_headers = []
    for header in headers:
        norm = _normalize_header(header)
        normalized = None
        for key, aliases in HEADER_ALIASES.items():
            if norm in aliases:
                normalized = key
                break
        if not normalized:
            normalized = norm
        mapping[header] = normalized
        if normalized not in normalized_headers:
            normalized_headers.append(normalized)

    return {
        "raw_headers_detected": headers,
        "normalized_headers": normalized_headers,
        "mapping": mapping,
    }


def _convert_number(value):
    """Convert numeric strings to ``int`` or ``float`` when possible."""

    if isinstance(value, str):
        value = value.replace(".", "").replace(",", ".") if value.count(",") else value
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value.strip()
    return value


def extract_products(document: str, headers: dict, knowledge_base):
    """Extract structured product data based on ``headers``."""

    import json

    normalized = json.dumps(headers.get("normalized_headers", []))
    rules = PRODUCT_ITEMS_PROMPT.replace("{TEXTO_BRUTO}", document).replace(
        "{NORMALIZED_HEADERS}", normalized
    )
    response = identify_field("products", "", rules, knowledge_base)

    try:
        data = json.loads(response)
    except Exception:
        data = {"products": [], "totals_detected": {}}

    for product in data.get("products", []):
        details = []
        seq_no = 1
        for detail in product.get("details", []):
            detail["seq_no"] = seq_no
            seq_no += 1
            details.append({k: _convert_number(v) for k, v in detail.items()})
        product["details"] = details

    totals = data.get("totals_detected", {})
    data["totals_detected"] = {k: _convert_number(v) for k, v in totals.items()}

    return data


def identify_products(document: str, knowledge_base):
    """High level helper that extracts headers and then products."""

    headers = extract_product_headers(document, knowledge_base)
    return extract_products(document, headers, knowledge_base)
