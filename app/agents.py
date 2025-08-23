import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    logger.info("Normalizing sales condition: %s", value)
    code = value.strip().upper()
    logger.debug("Normalized code: %s", code)
    if code in SALE_CONDITIONS:
        logger.info("Sales condition recognized: %s", code)
        return code
    logger.error("Unknown sales condition: %s", code)
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

    logger.info("Checking if exporter is also manufacturer (text length=%d)", len(text))
    normalized_text = " ".join(text.lower().split())
    logger.debug("Normalized text: %s", normalized_text)

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

    result = any(re.search(pattern, normalized_text) for pattern in patterns)
    logger.debug("Exporter-manufacturer match: %s", result)
    return result


def exportador_fabricante(text: str) -> bool:
    """Convenience wrapper to keep a Portuguese friendly API name."""
    logger.info("Calling exportador_fabricante")
    result = exporter_is_manufacturer(text)
    logger.info("exportador_fabricante result: %s", result)
    return result


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

    logger.info("Extracting possible company names")
    names = []
    for line in text.splitlines():
        stripped = line.strip()
        logger.debug("Processing line: %s", stripped)
        if not stripped or stripped.lower().startswith("to"):
            continue
        match = pattern.search(stripped)
        if match:
            name = match.group(1).strip()
            logger.debug("Found company name: %s", name)
            names.append(name)
    logger.info("Total company names found: %d", len(names))
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

    logger.info("Selecting relevant chunks for query: %s", query)
    scored = [
        (SequenceMatcher(None, query.lower(), chunk.lower()).ratio(), chunk)
        for chunk in chunks
    ]
    logger.debug("Scored %d chunks", len(scored))
    scored.sort(reverse=True)
    top = scored[:top_k]
    logger.debug("Top chunks: %s", top)
    return top


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

    logger.info("Identifying field '%s'", field_name)
    logger.debug("Rules: %s", rules)
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_API_MODEL")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set")
        raise EnvironmentError("OPENAI_API_KEY is not set")
    if not model:
        logger.error("OPENAI_API_MODEL is not set")
        raise EnvironmentError("OPENAI_API_MODEL is not set")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    query = f"{field_name} {rules}".strip()
    logger.debug("Query for relevant chunks: %s", query)
    relevant = select_relevant_chunks(query, knowledge_base)
    logger.debug("Retrieved %d relevant chunks", len(relevant))
    if not relevant or relevant[0][0] < 0.3:
        context = document
        logger.debug("Using full document as context")
    else:
        context = "\n---\n".join(chunk for _, chunk in relevant)
        logger.debug("Context built from relevant chunks")

    prompt = (
        f"{rules}\n\nContext:\n{context}\n\nDocument:\n{document}\n"
        f"Extract the value for '{field_name}'."
    )
    logger.debug("Prompt length: %d", len(prompt))

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content.strip()
    logger.info("Field '%s' identified as: %s", field_name, result)
    return result


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

    logger.info("Identifying multiple fields: %s", list(field_rules.keys()))
    results = {}
    for field_name, rules in field_rules.items():
        logger.debug("Processing field: %s", field_name)
        results[field_name] = identify_field(
            field_name, document, rules, knowledge_base
        )
        logger.debug("Result for %s: %s", field_name, results[field_name])
    logger.info("Completed field identification")
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

    logger.debug("Normalizing header: %s", value)
    normalized = unicodedata.normalize("NFD", value)
    without_accents = "".join(
        c for c in normalized if unicodedata.category(c) != "Mn"
    )
    result = without_accents.lower().strip()
    logger.debug("Normalized header result: %s", result)
    return result


def extract_product_headers(document: str, knowledge_base):
    """Identify product table headers and map them to normalized names."""

    import re
    logger.info("Extracting product headers")
    rules = PRODUCT_HEADERS_PROMPT.replace("{TEXTO_BRUTO}", document)
    response = identify_field("product_headers", "", rules, knowledge_base)
    logger.debug("LLM response for headers: %s", response)
    headers = [h.strip() for h in re.split(r"[,\n]", response) if h.strip()]

    mapping = {}
    normalized_headers = []
    for header in headers:
        logger.debug("Processing header: %s", header)
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
        logger.debug("Mapped '%s' to '%s'", header, normalized)

    result = {
        "raw_headers_detected": headers,
        "normalized_headers": normalized_headers,
        "mapping": mapping,
    }
    logger.info("Header extraction result: %s", result)
    return result


def _convert_number(value):
    """Convert numeric strings to ``int`` or ``float`` when possible."""
    logger.debug("Converting value: %s (%s)", value, type(value))
    if isinstance(value, str):
        value = value.replace(".", "").replace(",", ".") if value.count(",") else value
        try:
            if "." in value:
                converted = float(value)
            else:
                converted = int(value)
            logger.debug("Converted to numeric: %s", converted)
            return converted
        except ValueError:
            stripped = value.strip()
            logger.debug("Value not numeric, returning stripped: %s", stripped)
            return stripped
    logger.debug("Value not a string, returning as-is")
    return value


def extract_products(document: str, headers: dict, knowledge_base):
    """Extract structured product data based on ``headers``."""

    import json

    logger.info("Extracting products")
    normalized = json.dumps(headers.get("normalized_headers", []))
    logger.debug("Normalized headers JSON: %s", normalized)
    rules = PRODUCT_ITEMS_PROMPT.replace("{TEXTO_BRUTO}", document).replace(
        "{NORMALIZED_HEADERS}", normalized
    )
    response = identify_field("products", "", rules, knowledge_base)
    logger.debug("LLM response for products: %s", response)

    try:
        data = json.loads(response)
        logger.debug("Parsed JSON data: %s", data)
    except Exception as exc:
        logger.warning("Failed to parse products JSON: %s", exc)
        data = {"products": [], "totals_detected": {}}

    for product in data.get("products", []):
        logger.debug("Processing product: %s", product)
        details = []
        seq_no = 1
        for detail in product.get("details", []):
            detail["seq_no"] = seq_no
            seq_no += 1
            converted = {k: _convert_number(v) for k, v in detail.items()}
            logger.debug("Converted detail: %s", converted)
            details.append(converted)
        product["details"] = details

    totals = data.get("totals_detected", {})
    data["totals_detected"] = {k: _convert_number(v) for k, v in totals.items()}
    logger.info("Totals detected: %s", data.get("totals_detected"))

    return data


def identify_products(document: str, knowledge_base):
    """High level helper that extracts headers and then products."""
    logger.info("Identifying products from document")
    headers = extract_product_headers(document, knowledge_base)
    logger.debug("Headers extracted for products: %s", headers)
    products = extract_products(document, headers, knowledge_base)
    logger.debug("Products extracted: %s", products)
    return products
