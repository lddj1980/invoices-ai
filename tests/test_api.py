import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.main import app, vector_store, DEFAULT_PDF_PROMPT


PDF_WITH_TEXT = b"""%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n4 0 obj<</Length 55>>stream\nBT\n/F1 24 Tf\n50 150 Td\n(FOB sale condition: FOB) Tj\nET\nendstream\nendobj\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000079 00000 n \n0000000178 00000 n \n0000000401 00000 n \n0000000514 00000 n \ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n611\n%%EOF"""


def test_upload_invoice_returns_text():
    client = TestClient(app)
    response = client.post(
        '/invoice-text',
        files={'file': ('invoice.pdf', PDF_WITH_TEXT, 'application/pdf')},
    )
    assert response.status_code == 200
    body = response.json()
    assert 'FOB sale condition: FOB' in body['text']

def test_upload_invoice_extracts_fields(monkeypatch):
    client = TestClient(app)

    def fake_identify_fields(document, field_rules, knowledge_base):
        assert "FOB" in document
        assert knowledge_base  # persistence to vector store
        return {
            "sales_condition": "FOB",
            "exportador_fabricante": "não",
            "nome_do_exportador": "ACME LTDA",
        }

    def fake_extract_product_headers(document, knowledge_base):
        return {
            "raw_headers_detected": [],
            "normalized_headers": [],
            "mapping": {},
        }

    def fake_extract_products(document, headers, knowledge_base):
        return {
            "products": [
                {"no": 1, "description": "ABC", "details": []},
                {"no": 2, "description": "DEF", "details": []},
            ],
            "totals_detected": {},
        }

    vector_store.chunks = []
    monkeypatch.setattr("app.main.identify_fields", fake_identify_fields)
    monkeypatch.setattr(
        "app.main.extract_product_headers", fake_extract_product_headers
    )
    monkeypatch.setattr("app.main.extract_products", fake_extract_products)

    response = client.post(
        "/invoices", files={"file": ("invoice.pdf", PDF_WITH_TEXT, "application/pdf")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["fields"]["sales_condition"] == "FOB"
    assert body["fields"]["exportador_fabricante"] == "não"
    assert body["fields"]["nome_do_exportador"] == "ACME LTDA"
    assert body["fields"]["produtos"]["products"][0]["description"] == "ABC"
    assert vector_store.chunks  # ensure document was added


def test_upload_invoice_uses_ocr_when_no_text(monkeypatch):
    client = TestClient(app)

    blank_pdf = b"%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000079 00000 n \n0000000135 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"

    class DummyPdfReader:
        def __init__(self, *a, **k):
            self.pages = [SimpleNamespace(extract_text=lambda: "")]

    class DummyOCR:
        def __init__(self, *a, **k):
            pass

        def ocr(self, path):
            return [[[[[0, 0], [1, 0], [1, 1], [0, 1]], ("FOB sale condition: FOB", 0.9)]]]

    vector_store.chunks = []
    monkeypatch.setattr(
        "app.main.identify_fields", lambda *args, **kwargs: {"sales_condition": "FOB"}
    )
    monkeypatch.setattr(
        "app.main.extract_product_headers",
        lambda *args, **kwargs: {"raw_headers_detected": [], "normalized_headers": [], "mapping": {}},
    )
    monkeypatch.setattr(
        "app.main.extract_products",
        lambda *args, **kwargs: {"products": [], "totals_detected": {}},
    )
    monkeypatch.setattr("app.main.PdfReader", DummyPdfReader)
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=lambda *a, **k: DummyOCR()),
    )

    response = client.post(
        "/invoices", files={"file": ("invoice.pdf", blank_pdf, "application/pdf")}
    )
    assert response.status_code == 200
    assert vector_store.chunks  # document added even via OCR
    assert any("FOB" in chunk for chunk in vector_store.chunks)


def test_upload_invoice_uses_image_ocr(monkeypatch):
    """Ensure image uploads trigger the image OCR pathway."""
    client = TestClient(app)

    # Fake image bytes; actual content is irrelevant because OCR is monkeypatched.
    image_bytes = b"notreallyanimage"

    def fake_image_ocr(data: bytes) -> str:
        assert data == image_bytes
        return "FOB sale condition: FOB"

    monkeypatch.setattr("app.main.image_to_text", fake_image_ocr)

    response = client.post(
        "/invoice-text",
        files={"file": ("invoice.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "FOB sale condition" in body["text"]


def test_upload_invoice_handles_env_error(monkeypatch):
    client = TestClient(app)

    def fake_identify_fields(document, field_rules, knowledge_base):
        return {}

    def fake_extract_product_headers(document, knowledge_base):
        return {"raw_headers_detected": [], "normalized_headers": [], "mapping": {}}

    def raise_env_error(*args, **kwargs):
        raise EnvironmentError("OPENAI_API_KEY is not set")

    vector_store.chunks = []
    monkeypatch.setattr("app.main.identify_fields", fake_identify_fields)
    monkeypatch.setattr("app.main.extract_product_headers", fake_extract_product_headers)
    monkeypatch.setattr("app.main.extract_products", raise_env_error)

    response = client.post(
        "/invoices", files={"file": ("invoice.pdf", PDF_WITH_TEXT, "application/pdf")}
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "OPENAI_API_KEY is not set"


def test_invoices_gpt_pdf_accepts_custom_prompt(monkeypatch):
    client = TestClient(app)

    class DummyClient:
        def __init__(self, *a, **k):
            self.files = SimpleNamespace(create=lambda file, purpose: SimpleNamespace(id="fid"))

            def create_response(model, input):
                assert input[0]["content"][0]["text"] == "custom"
                return SimpleNamespace(output_text="ok")

            self.responses = SimpleNamespace(create=create_response)

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda api_key: DummyClient()))

    response = client.post(
        "/invoices-gpt-pdf",
        data={"prompt": "custom"},
        files={"file": ("invoice.pdf", PDF_WITH_TEXT, "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["file_id"] == "fid"
    assert body["result"] == "ok"


def test_invoices_gpt_pdf_uses_default_prompt(monkeypatch):
    client = TestClient(app)

    class DummyClient:
        def __init__(self, *a, **k):
            self.files = SimpleNamespace(create=lambda file, purpose: SimpleNamespace(id="fid"))

            def create_response(model, input):
                assert input[0]["content"][0]["text"] == DEFAULT_PDF_PROMPT
                return SimpleNamespace(output_text="ok")

            self.responses = SimpleNamespace(create=create_response)

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda api_key: DummyClient()))

    response = client.post(
        "/invoices-gpt-pdf",
        files={"file": ("invoice.pdf", PDF_WITH_TEXT, "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["file_id"] == "fid"
    assert body["result"] == "ok"
