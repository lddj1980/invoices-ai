import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.main import app, vector_store


def test_upload_invoice_extracts_fields(monkeypatch):
    client = TestClient(app)

    def fake_identify_field(field_name, document, rules, knowledge_base):
        assert field_name == "sales_condition"
        assert "FOB" in document
        assert knowledge_base  # persistence to vector store
        return "FOB"

    vector_store.chunks = []
    monkeypatch.setattr("app.main.identify_field", fake_identify_field)

    response = client.post(
        "/invoices", files={"file": ("invoice.txt", b"FOB sale condition: FOB")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["fields"]["sales_condition"] == "FOB"
    assert vector_store.chunks  # ensure document was added
