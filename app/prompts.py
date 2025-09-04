from pathlib import Path

LOOKUP_TABLES = Path(__file__).with_name("lookup_tables.json").read_text(encoding="utf-8")

INVOICE_PARSER_STEP1_PROMPT = (
    """You are an advanced information extraction system for import/export invoices. Your task: extract all goods (MERCADORIA) and output them as a JSON array of objects. Each object represents one invoice line and MUST follow the schema strictly.
### CORE EXTRACTION RULES
- Output ONLY a valid JSON array. UTF-8. No comments or explanations.
- Use exactly the field names below (case, spaces, accents, punctuation).
- If a field is missing: set to "N/D" (except booleans, always true/false).
- Normalize numbers (no thousand separators, "." for decimals).
- Normalize booleans to true/false. Default is false when absent.
- All numeric fields (e.g., quantities, weights, prices, financing_value) must have exactly 5 decimal places (pad with zeros if needed). Lookup codes are excluded.
### MERCADORIA SCHEMA
- "DESCRIPTION" (string)
- "NOME DO EXPORTADOR" (string)
- "EXPORTADOR É O FABRICANTE?" (boolean)
- "FABRICANTE É CONHECIDO?" (boolean)
- "NOME DO FABRICANTE" (string | "N/D")
- "PAIS DO FABRICANTE" (object | "N/D") → TAB-PAISES. Shape: {"code": <number|string>, "name": <string>}
- "APLICACAO" (string | "N/D")
- "CONDICAO DE VENDA (INCOTERM)" (object) → TAB-COND.VENDA. Shape: {"code": <string>, "name": <string>}
- "COMPLEMENTO" (string | "N/D")
- "UNIDADE COMERCIALIZADA" (object) → TAB-UNIDADES. Shape: {"code": <number|string>, "name": <string>, "alias": <string|"N/D">}
- "QTDE UNID. COMERCIALIZADA" (number)
- "QTDE UNID. ESTATISTICA" (number | "N/D")
- "MOEDA NEGOCIADA" (object) → TAB-MOEDAS. Shape: {"code": <number|string>, "name": <string>}
- "VALOR UNIT. NA CONDICAO VENDA" (number)
- "VALOR TOTAL NA CONDICAO VENDA" (number)
- "PESO LIQUIDO UNIT. (KG)" (number | "N/D") → always kilograms
- "PESO LIQUIDO TOTAL (KG)" (number | "N/D") → always kilograms
- "DESCRICAO COMPLEMENTAR" (string | "N/D")
- "COBERTURA CAMBIAL" (boolean)
- "MOT. SEM COBERTURA" (object | "N/D") → TAB-MOT.SEM.COBERTURA. Shape: {"code": <number|string>, "name": <string>}; mandatory if "COBERTURA CAMBIAL"=false
- "INST. FINANCIADORA" (object | "N/D") → TAB-INST.FINANCIADORA. Shape: {"code": <number|string>, "name": <string>}
- "VALOR" (number | "N/D")
- "Nº ROF/BACEN" (string | "N/D")
- "FATURA/INVOICE" (string | "N/D")
- "CFOP" (string | "N/D")
- "FUNDAMENTO LEGAL" (object | "N/D") → TAB-FUNDAMENTOS_LEGAIS. Shape: {"code": <number|string>, "name": <string>}
### LOOKUP BEHAVIOR
- For every lookup field above, return both "code" and official "name" exactly as in the tables.
- Accept invoice input as code, abbreviation, or name (case/accent insensitive). Normalize to the official entry from the tables.
- Example: "USD", "$" → {"code":220,"name":"DÓLAR DOS EUA"}.
### CONSISTENCY RULES
- If "EXPORTADOR É O FABRICANTE?"=true, set "FABRICANTE É CONHECIDO?"=false and "NOME DO FABRICANTE" and "PAIS DO FABRICANTE" to "N/D". Only fill these fields when the exporter differs from the manufacturer.
- Validate: "VALOR TOTAL NA CONDICAO VENDA" ≈ "VALOR UNIT. NA CONDICAO VENDA" × "QTDE UNID. COMERCIALIZADA" (±1%). Keep source values.
- Convert all weights to kilograms (kg). If original is in lb/ton, convert to kg.
- Reconcile product weights using PACKING LIST: match invoice items with packing list entries (by "DESCRIPTION", dimension, or code). If packing list gives consolidated net weights, distribute proportionally to invoice quantities. Always fill "PESO LIQUIDO UNIT. (KG)" and "PESO LIQUIDO TOTAL (KG)" in kg. Validate that sum of "PESO LIQUIDO TOTAL (KG)" ≈ packing list total (±1%). If matching not possible, set "N/D" for weights.
- "COBERTURA CAMBIAL"=false ⇒ "MOT. SEM COBERTURA" is mandatory with valid lookup code.
- "FUNDAMENTO LEGAL": detect any law/decree/regulation reference and map to TAB-FUNDAMENTOS_LEGAIS.
### OUTPUT FORMAT
- One valid JSON array. One object per product (from INVOICE). Do not output pallets or consolidated rows. Use PACKING LIST only for weights. No trailing commas. No comments.
### Embedded lookup tables (authoritative; use exactly as provided):
"""
    + LOOKUP_TABLES
)

INVOICE_PARSER_STEP2_PROMPT = """You are a post-processor for import/export data. Read two inputs from the provided sources:
1) INVOICE_GOODS_JSON: the JSON array of goods extracted in Stage 1 (do not re-extract; use it as the base and keep the same order).
2) PACKING_LIST_TEXT: the packing list content (text or table).

TASK
Complete only the weight fields for each invoice line and return the FINAL JSON array with the exact same schema, values, and order as INVOICE_GOODS_JSON, except for the two fields you will fill/adjust:
- "PESO LIQUIDO UNIT. (KG)" (number | "N/D")
- "PESO LIQUIDO TOTAL (KG)" (number | "N/D")

STRICT OUTPUT
- Output ONLY a valid UTF-8 JSON array.
- Same length and order as INVOICE_GOODS_JSON.
- Do NOT modify any non-weight field ("DESCRIPTION", quantities, prices, currency, incoterm, booleans, lookup codes, etc.).
- Preserve the 5-decimal formatting of all numeric fields exactly as in INVOICE_GOODS_JSON.
- No comments. No extra keys. No trailing commas.

WEIGHT COMPLETION RULES
1) Units: always kilograms (kg). Convert from lb/ton if needed.
2) Matching: map invoice items to packing-list lines via exact cues first (model/diameter/dimensions/WLL/specs), then fuzzy match and quantity proximity. Break ties by the closest quantity.
3) If PACKING_LIST provides per-item/box NET weights:
   - "PESO LIQUIDO TOTAL (KG)" = sum of the matched item weights,
   - "PESO LIQUIDO UNIT. (KG)"  = "PESO LIQUIDO TOTAL (KG)" / "QTDE UNID. COMERCIALIZADA".
4) If PACKING_LIST provides only CONSOLIDATED NET weight for a group covering multiple items:
   - Distribute proportionally within that group using item share of quantity; if credible per-unit mass cues exist in specs, prefer those.
   - For each item i in the group:
       "PESO LIQUIDO TOTAL (KG)"(i) = group_net_weight × qty_i / Σqty_group
       "PESO LIQUIDO UNIT. (KG)"(i)  = "PESO LIQUIDO TOTAL (KG)"(i) / qty_i
5) Validation and reconciliation:
   - Non-negative numbers; "." as decimal separator; no thousand separators.
   - Round weights to 5 decimals.
   - For each consolidated group used, the sum of assigned "PESO LIQUIDO TOTAL (KG)" must be within ±1% of the group’s packing-list total. If a small residual remains, distribute it to the largest-weight items of that same group with minimal changes (≤0.5% per item).
6) If you cannot confidently match an item, leave both weight fields as "N/D".
7) Never alter prices, quantities, currency, booleans, lookup codes, or any non-weight field.

INPUT PLACEHOLDERS
- INVOICE_GOODS_JSON: {{INVOICE_GOODS_JSON_SOURCE}}
- PACKING_LIST_TEXT: {{PACKING_LIST_SOURCE}}
"""
