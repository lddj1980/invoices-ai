# invoices-ai

## Condições de venda suportadas

O sistema reconhece as seguintes [Condições de venda](#condições-de-venda-suportadas):

| Código | Nome completo |
| ------ | ------------- |
| CFR | Cost and Freight |
| CIF | Cost, Insurance and Freight |
| CIP | Carriage and Insurance Paid To |
| CPT | Carriage Paid To |
| C_F | Custo e Frete |
| C_I | Custo e Seguro |
| DAP | Delivered At Place |
| DAT | Delivered At Terminal |
| EXW | Ex Works |
| FAS | Free Alongside Ship |
| FCA | Free Carrier |
| FOB | Free On Board |
| OCV | Outra [Condição de venda](#condições-de-venda-suportadas) |

### Identificação e validação

O agente solicita uma [Condição de venda](#condições-de-venda-suportadas) por meio de um prompt que lista todos os códigos válidos.
A função `normalize_sales_condition` converte o valor informado para maiúsculas, remove espaços e confirma que o código existe em `SALE_CONDITIONS`.
Se a [Condição de venda](#condições-de-venda-suportadas) não for reconhecida, o agente gera um erro orientando a consulta desta seção.

## Extração de produtos

O fluxo de extração de itens utiliza **dois agentes**:

1. `extract_product_headers` – usa o *Prompt 1* para localizar e normalizar os cabeçalhos da tabela de produtos.
2. `extract_products` – com os cabeçalhos normalizados e o *Prompt 2*, extrai os itens e totais da fatura.

Exemplo de retorno:

```json
{
  "products": [
    {
      "no": "1",
      "description": "Item A",
      "details": [
        {
          "seq_no": 1,
          "size_spec": "M",
          "qty_pc": 10,
          "net_weight_kg": 5.0
        }
      ]
    }
  ],
  "totals_detected": {"total_net_weight_kg": 5.0}
}
```

Os prompts utilizados são definidos em `app/agents.py` nas constantes `PRODUCT_HEADERS_PROMPT` e `PRODUCT_ITEMS_PROMPT`.
