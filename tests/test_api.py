import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.main import app, vector_store


PDF_WITH_TEXT = b"""%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n4 0 obj<</Length 55>>stream\nBT\n/F1 24 Tf\n50 150 Td\n(FOB sale condition: FOB) Tj\nET\nendstream\nendobj\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000079 00000 n \n0000000178 00000 n \n0000000401 00000 n \n0000000514 00000 n \ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n611\n%%EOF"""


def test_upload_invoice_extracts_fields(monkeypatch):
    client = TestClient(app)

    def fake_identify_field(field_name, document, rules, knowledge_base):
        assert "FOB" in document
        assert knowledge_base  # persistence to vector store
        if field_name == "sales_condition":
            return "FOB"
        if field_name == "exportador_fabricante":
            return "não"
        raise AssertionError("unexpected field")

    vector_store.chunks = []
    monkeypatch.setattr("app.main.identify_field", fake_identify_field)

    response = client.post(
        "/invoices", files={"file": ("invoice.pdf", PDF_WITH_TEXT, "application/pdf")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["fields"]["sales_condition"] == "FOB"
    assert body["fields"]["exportador_fabricante"] == "não"
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
    monkeypatch.setattr("app.main.identify_field", lambda *args, **kwargs: "FOB")
    monkeypatch.setattr("app.main.PdfReader", DummyPdfReader)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=lambda *a, **k: DummyOCR()))

    response = client.post(
        "/invoices", files={"file": ("invoice.pdf", blank_pdf, "application/pdf")}
    )
    assert response.status_code == 200
    assert vector_store.chunks  # document added even via OCR
    assert any("FOB" in chunk for chunk in vector_store.chunks)
