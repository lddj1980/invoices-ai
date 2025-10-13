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
- "ITEM" (string)
- "PARTNUMBER" (string) → "descrição do produto+size"
- "NOME DO EXPORTADOR" (string)
- "EXPORTADOR É O FABRICANTE?" (boolean)
- "FABRICANTE É CONHECIDO?" (boolean)
- "NOME DO FABRICANTE" (string | "N/D")
- "PAIS DO FABRICANTE" (object | "N/D") → TAB-PAISES. Shape: {"code": <string>}
- "APLICACAO" (string | "N/D")
- "CONDICAO DE VENDA (INCOTERM)" (object) → TAB-COND.VENDA. Shape: {"code": <string>}
- "COMPLEMENTO" (string | "N/D") → inclua informações sobre condições relacionadas ao frete, quando disponíveis na invoice
- "UNIDADE COMERCIALIZADA" (object) → TAB-UNIDADES. Shape: {"code": <string>}
- "QTDE UNID. COMERCIALIZADA" (number)
- "QTDE UNID. ESTATISTICA" (number | "N/D")
- "MOEDA NEGOCIADA" (object) → TAB-MOEDAS. Shape: {"code": <string>}
- "VALOR UNIT. NA CONDICAO VENDA" (number)
- "VALOR TOTAL NA CONDICAO VENDA" (number)
- "PESO LIQUIDO UNIT. (KG)" (number | "N/D") → always kilograms
- "PESO LIQUIDO TOTAL (KG)" (number | "N/D") → always kilograms
 - "DESCRICAO COMPLEMENTAR" (string | "N/D")
 - "COBERTURA CAMBIAL" (string) → default "0000_A_0180DIAS" unless the PACKING LIST clearly indicates a payment term shorter than 180 days; in that case use the corresponding coverage code (e.g., "0000_A_0090DIAS" for payment in 90 days)
 - "MOT. SEM COBERTURA" (object | "N/D") → TAB-MOT.SEM.COBERTURA. Shape: {"code": <string>}; mandatory if "COBERTURA CAMBIAL" indicates no coverage
 - "INST. FINANCIADORA" (object | "N/D") → TAB-INST.FINANCIADORA. Shape: {"code": <string>}
- "VALOR" (number | "N/D")
- "Nº ROF/BACEN" (string | "N/D")
- "FATURA/INVOICE" (string | "N/D")
- "CFOP" (string | "N/D")
- "FUNDAMENTO LEGAL" (object | "N/D") → TAB-FUNDAMENTOS_LEGAIS. Shape: {"code": <string>}
### LOOKUP BEHAVIOR
- For every lookup field above, output only the lookup "code" (string) validated against the tables.
- Accept invoice input as code, abbreviation, or name (case/accent insensitive). Normalize to the official code from the tables.
- Example: "USD", "$" → {"code":"220"}.
### IDENTIFICAÇÃO DO EXPORTADOR E DO FABRICANTE
- O exportador é a parte que vende ou envia os produtos. Procure cabeçalhos ou rótulos como "Exporter", "Exportador", "Supplier" ou "Seller". Não confunda com o consignatário ou importador.
- O fabricante é a parte que produziu os produtos. Procure termos como "Manufacturer", "Fabricante", "Made by" ou "Produced by".
- Se a invoice indicar explicitamente que o exportador também é o fabricante, marque "EXPORTADOR É O FABRICANTE?" = true.
### CONSISTENCY RULES
- If "EXPORTADOR É O FABRICANTE?"=true, set "FABRICANTE É CONHECIDO?"=false and "NOME DO FABRICANTE" and "PAIS DO FABRICANTE" to "N/D". Only fill these fields when the exporter differs from the manufacturer.
- Se "EXPORTADOR É O FABRICANTE?" = false, procure explicitamente na invoice o "NOME DO FABRICANTE" e o "PAIS DO FABRICANTE". Se encontrados, preencha os campos e marque "FABRICANTE É CONHECIDO?" = true; caso contrário, use "N/D" e "FABRICANTE É CONHECIDO?" = false.
- Validate: "VALOR TOTAL NA CONDICAO VENDA" ≈ "VALOR UNIT. NA CONDICAO VENDA" × "QTDE UNID. COMERCIALIZADA" (±1%). Keep source values.
- Convert all weights to kilograms (kg). If original is in lb/ton, convert to kg.
- Reconcile product weights using PACKING LIST: match invoice items with packing list entries (by "PARTNUMBER"/"ITEM", dimension, or code). If packing list gives consolidated net weights, distribute proportionally to invoice quantities. Always fill "PESO LIQUIDO UNIT. (KG)" and "PESO LIQUIDO TOTAL (KG)" in kg. Validate that sum of "PESO LIQUIDO TOTAL (KG)" ≈ packing list total (±1%). If matching not possible, set "N/D" for weights.
 - Default "COBERTURA CAMBIAL" to "0000_A_0180DIAS" unless the PACKING LIST specifies a payment term shorter than 180 days; then select the matching coverage code.
 - If "COBERTURA CAMBIAL" denotes no coverage, "MOT. SEM COBERTURA" is mandatory with valid lookup code.
 - "FUNDAMENTO LEGAL": detect any law/decree/regulation reference and map to TAB-FUNDAMENTOS_LEGAIS.
### OUTPUT FORMAT
- One valid JSON array. One object per product (from INVOICE). Do not output pallets or consolidated rows. Use PACKING LIST only for weights. No trailing commas. No comments.
### Embedded lookup tables (authoritative; use exactly as provided):
"""
    + LOOKUP_TABLES
)


INVOICE_PARSER_STEP2_PROMPT = """
You are a post-processor for import/export data. Read two inputs from the provided sources:
1) INVOICE_GOODS_JSON: the JSON array of goods extracted in Stage 1 (do not re-extract; use it as the base and keep the same order).
2) PACKING_LIST_TEXT: the packing list content (text or table).

TASK
Complete only the weight fields for each invoice line and return the FINAL JSON array with the exact same schema, values, and order as INVOICE_GOODS_JSON, except for the two fields you will fill/adjust:
- "PESO LIQUIDO UNIT. (KG)" (number | "N/D")
- "PESO LIQUIDO TOTAL (KG)" (number | "N/D")

STRICT OUTPUT
- Output ONLY a valid UTF-8 JSON array.
- Same length and order as INVOICE_GOODS_JSON.
- Do NOT modify any non-weight field ("ITEM", "PARTNUMBER", quantities, prices, currency, incoterm, booleans, lookup codes, etc.).
- Preserve the 5-decimal formatting of all numeric fields exactly as in INVOICE_GOODS_JSON.
- No comments. No extra keys. No trailing commas.

WEIGHT COMPLETION RULES
1) Units: always kilograms (kg). Convert from lb/ton if needed.
2) Matching: map invoice items to packing-list lines via PARTNUMBER/ITEM and exact cues first (model/diameter/dimensions/WLL/specs), then fuzzy match and quantity proximity. Break ties by the closest quantity.
3) Product-Scoped Grouping (critical):
   - The packing list is structured by pallets, and inside each pallet by product families (e.g., "MASTER LINK A342", "G80 LIFTING SCREW POINT").
   - A block of NET/GROSS weight belongs exclusively to the product family under its header. 
   - Rule PL-1: A product family header is a line with the family name (often uppercase) followed or preceded by N.W./G.W. values.
   - Rule PL-2: The scope of those N.W./G.W. values ends at the next family header or the end of the pallet.
   - Rule PL-3: Variants (different sizes, diameters, WLLs) listed under the same family inherit that family’s NET weight block.
   - Rule PL-4: Never propagate NET/GROSS weights across different families, even if they appear in the same pallet. 
     Example: If pallet P4-28 shows "MASTER LINK A342" with N.W. 983.00/G.W. 1,044.00, those weights apply only to the MASTER LINK A342 items. 
     Items like "G80 LIFTING SCREW POINT" on the same pallet must remain "N/D" unless they have their own weight block.

4) If PACKING_LIST provides per-item/box NET weights:
   - "PESO LIQUIDO TOTAL (KG)" = sum of the matched item weights,
   - "PESO LIQUIDO UNIT. (KG)"  = "PESO LIQUIDO TOTAL (KG)" / "QTDE UNID. COMERCIALIZADA".
5) If PACKING_LIST provides only CONSOLIDATED NET weight for a product family:
   - Distribute proportionally among its invoice items using quantity share. 
   - If per-unit mass cues exist in specs, prefer those over proportional distribution.
   - For each item i in the family:
       "PESO LIQUIDO UNIT. (KG)"(i)  = family_net_weight / Σqty_family
       "PESO LIQUIDO TOTAL (KG)"(i) = "PESO LIQUIDO UNIT. (KG)"(i) × qty_i
6) Validation and reconciliation:
   - Non-negative numbers; "." as decimal separator; no thousand separators.
   - Round weights to 5 decimals.
   - For each consolidated family used, the sum of assigned "PESO LIQUIDO TOTAL (KG)" must be within ±1% of the family’s packing-list total. 
   - If a small residual remains, distribute it to the largest-weight items of that same family with minimal changes (≤0.5% per item).
7) If you cannot confidently match an item, or if the item’s family has no weight block in the packing list, leave both weight fields as "N/D".
8) Never alter prices, quantities, currency, booleans, lookup codes, or any non-weight field.

FALLBACKS
- Ambiguity or conflicting matches ⇒ "N/D".
- Normalize decimal notation (use "."; remove thousand separators).
- Ignore case, pluralization, or punctuation differences in product family names.

INPUT PLACEHOLDERS
- INVOICE_GOODS_JSON: {{INVOICE_GOODS_JSON_SOURCE}}
- PACKING_LIST_TEXT: {{PACKING_LIST_SOURCE}}
"""
