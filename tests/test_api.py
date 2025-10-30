import sys
from pathlib import Path
from types import SimpleNamespace
import json
import io

from PyPDF2 import PdfReader

import pytest
from fastapi.testclient import TestClient

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.main import app, vector_store, DEFAULT_PDF_PROMPT, jobs, _finalize_job
from app.prompts import (
    INVOICE_PARSER_STEP1_PROMPT,
    INVOICE_PARSER_STEP2_PROMPT,
)


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
                {"no": 1, "item": "", "partnumber": "ABC", "details": []},
                {"no": 2, "item": "", "partnumber": "DEF", "details": []},
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
    assert body["fields"]["produtos"]["products"][0]["partnumber"] == "ABC"
    assert body["fields"]["produtos"]["products"][0]["item"] == ""
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
            self.created = []
            self.deleted = []

            def create_file(file, purpose):
                idx = len(self.created) + 1
                fid = f"fid{idx}"
                self.created.append(fid)
                return SimpleNamespace(id=fid)

            def delete_file(fid):
                self.deleted.append(fid)

            def create_response(model, input):
                assert input[0]["role"] == "system"
                assert input[0]["content"] == "custom"
                assert input[1]["role"] == "user"
                content = input[1]["content"]
                assert content[0] == {"type": "input_file", "file_id": "fid2"}
                assert content[1] == {"type": "input_file", "file_id": "fid1"}
                return SimpleNamespace(output_text="ok")

            self.files = SimpleNamespace(create=create_file, delete=delete_file)
            self.responses = SimpleNamespace(create=create_response)

    dummy_client = DummyClient()
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda api_key, **kwargs: dummy_client),
    )

    response = client.post(
        "/invoices-gpt-pdf",
        data={"prompt": "custom"},
        files={
            "packing_list": ("packing.pdf", PDF_WITH_TEXT, "application/pdf"),
            "invoice": ("invoice.pdf", PDF_WITH_TEXT, "application/pdf"),
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    response = client.get(f"/invoices-gpt-pdf/{job_id}")
    body = response.json()
    assert body["result"]["file_ids"] == {"packing_list": "fid1", "invoice": "fid2"}
    assert body["result"]["result"] == "ok"
    assert dummy_client.deleted == ["fid1", "fid2"]


def test_invoice_json_gpt_pdf_converts_json(monkeypatch):
    client = TestClient(app)
    captured: dict[str, bytes] = {}

    def fake_process(job_id, packing_contents, invoice_contents, prompt):
        captured["invoice"] = invoice_contents
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = "ok"

    monkeypatch.setattr("app.main._process_invoices_gpt_pdf", fake_process)

    invoice_data = {"item": "ABC", "qty": 1}
    response = client.post(
        "/invoice-json-gpt-pdf",
        data={"invoice": json.dumps(invoice_data)},
        files={"packing_list": ("packing.pdf", PDF_WITH_TEXT, "application/pdf")},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    text = "".join(
        page.extract_text() or "" for page in PdfReader(io.BytesIO(captured["invoice"])).pages
    )
    assert "\"item\": \"ABC\"" in text

    body = client.get(f"/invoices-gpt-pdf/{job_id}").json()
    assert body["status"] == "done"
    assert body["result"] == "ok"


def test_invoice_json_gpt_pdf_invalid_json():
    client = TestClient(app)
    response = client.post(
        "/invoice-json-gpt-pdf",
        data={"invoice": "not-json"},
        files={"packing_list": ("packing.pdf", PDF_WITH_TEXT, "application/pdf")},
    )
    assert response.status_code == 400


def test_invoices_gpt_pdf_uses_default_prompt(monkeypatch):
    client = TestClient(app)

    class DummyClient:
        def __init__(self, *a, **k):
            self.created = []
            self.deleted = []

            def create_file(file, purpose):
                idx = len(self.created) + 1
                fid = f"fid{idx}"
                self.created.append(fid)
                return SimpleNamespace(id=fid)

            def delete_file(fid):
                self.deleted.append(fid)

            def create_response(model, input):
                assert input[0]["role"] == "system"
                assert input[0]["content"] == DEFAULT_PDF_PROMPT
                assert input[1]["role"] == "user"
                content = input[1]["content"]
                assert content[0] == {"type": "input_file", "file_id": "fid2"}
                assert content[1] == {"type": "input_file", "file_id": "fid1"}
                return SimpleNamespace(output_text="ok")

            self.files = SimpleNamespace(create=create_file, delete=delete_file)
            self.responses = SimpleNamespace(create=create_response)

    dummy_client = DummyClient()
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda api_key, **kwargs: dummy_client),
    )

    response = client.post(
        "/invoices-gpt-pdf",
        files={
            "packing_list": ("packing.pdf", PDF_WITH_TEXT, "application/pdf"),
            "invoice": ("invoice.pdf", PDF_WITH_TEXT, "application/pdf"),
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    response = client.get(f"/invoices-gpt-pdf/{job_id}")
    body = response.json()
    assert body["result"]["file_ids"] == {"packing_list": "fid1", "invoice": "fid2"}
    assert body["result"]["result"] == "ok"
    assert dummy_client.deleted == ["fid1", "fid2"]


def test_invoice_parser_pipeline(monkeypatch):
    client = TestClient(app)

    calls = []

    def fake_process(job_id, packing_contents, invoice_contents, prompt):
        calls.append((job_id, prompt, invoice_contents))
        if prompt == INVOICE_PARSER_STEP1_PROMPT:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = {"result": '[{"item":"","partnumber":"A","qty_commercial":1}]'}
        elif prompt == INVOICE_PARSER_STEP2_PROMPT:
            assert invoice_contents == b"PDF_FROM_JSON"
            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = "final"

    monkeypatch.setattr("app.main._process_invoices_gpt_pdf", fake_process)

    def fake_json_to_pdf(data):
        assert data == [{"item": "", "partnumber": "A", "qty_commercial": 1}]
        return b"PDF_FROM_JSON"

    monkeypatch.setattr("app.main._json_to_pdf", fake_json_to_pdf)

    response = client.post(
        "/invoice-parser",
        files={
            "packing_list": ("packing.pdf", PDF_WITH_TEXT, "application/pdf"),
            "invoice": ("invoice.pdf", PDF_WITH_TEXT, "application/pdf"),
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    body = client.get(f"/invoice-parser/{job_id}").json()
    assert body["status"] == "done"
    assert body["result"] == "final"
    assert calls[0][1] == INVOICE_PARSER_STEP1_PROMPT
    assert calls[1][1] == INVOICE_PARSER_STEP2_PROMPT


def test_invoices_gpt_pdf_triggers_webhook_on_success(monkeypatch):
    client = TestClient(app)
    jobs.clear()

    dispatched = []

    def capture_webhook(payload):
        dispatched.append(payload)

    monkeypatch.setattr("app.main._post_webhook", capture_webhook)

    def fake_process(job_id, packing_contents, invoice_contents, prompt):
        _finalize_job(job_id, "done", [{"PARTNUMBER": "ABC123"}])

    monkeypatch.setattr("app.main._process_invoices_gpt_pdf", fake_process)

    response = client.post(
        "/invoices-gpt-pdf",
        data={"external_job_id": "ext-123"},
        files={
            "packing_list": ("packing.pdf", PDF_WITH_TEXT, "application/pdf"),
            "invoice": ("invoice.pdf", PDF_WITH_TEXT, "application/pdf"),
        },
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert jobs[job_id]["status"] == "done"
    assert dispatched == [
        {
            "external_job_id": "ext-123",
            "status": "completed",
            "result": [{"PARTNUMBER": "ABC123"}],
        }
    ]


def test_invoices_gpt_pdf_triggers_webhook_on_error(monkeypatch):
    client = TestClient(app)
    jobs.clear()

    dispatched = []

    def capture_webhook(payload):
        dispatched.append(payload)

    monkeypatch.setattr("app.main._post_webhook", capture_webhook)

    def fake_process(job_id, packing_contents, invoice_contents, prompt):
        _finalize_job(job_id, "error", "processing failed")

    monkeypatch.setattr("app.main._process_invoices_gpt_pdf", fake_process)

    response = client.post(
        "/invoices-gpt-pdf",
        data={"external_job_id": "ext-err"},
        files={
            "packing_list": ("packing.pdf", PDF_WITH_TEXT, "application/pdf"),
            "invoice": ("invoice.pdf", PDF_WITH_TEXT, "application/pdf"),
        },
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert jobs[job_id]["status"] == "error"
    assert dispatched == [
        {
            "external_job_id": "ext-err",
            "status": "error",
            "error_message": "processing failed",
        }
    ]
