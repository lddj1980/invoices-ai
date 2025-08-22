import io
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Dict, List

from .agents import (
    identify_fields,
    select_relevant_chunks,
    extract_product_headers,
    extract_products,
)

try:  # pragma: no cover - handled via tests
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

app = FastAPI()


def _split_into_chunks(text: str, chunk_size: int = 200) -> List[str]:
    """Split ``text`` into chunks of up to ``chunk_size`` characters."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


class InMemoryVectorStore:
    """Very small in-memory vector store for demo purposes.

    The store keeps raw text chunks and uses ``select_relevant_chunks`` for
    similarity search. This mimics a vector database without external
    dependencies so tests remain lightweight.
    """

    def __init__(self) -> None:
        self.chunks: List[str] = []

    def add_document(self, text: str) -> None:
        self.chunks.extend(_split_into_chunks(text))

    def search(self, query: str, top_k: int = 3) -> List[str]:
        return [chunk for _, chunk in select_relevant_chunks(query, self.chunks, top_k=top_k)]


vector_store = InMemoryVectorStore()

FIELD_RULES: Dict[str, str] = {
    "sales_condition": "Identify the sales condition code present in the invoice.",
    "exportador_fabricante": "Determine se o exportador também é o fabricante. Responda 'sim' ou 'não'.",
    "nome_do_exportador": (
        "Liste possíveis nomes de empresas exportadoras presentes no documento que não estejam "
        "precedidos por palavras como 'To'. Responda apenas com o nome do possível exportador, sem "
        "justificar ou explicar."
    ),
}

def _extract_text(content: bytes, filename: str) -> str:
    """Extract text from ``content`` using PDF parsing or OCR.

    For PDFs the function first attempts :class:`PyPDF2.PdfReader`. When no text
    is found it falls back to :mod:`paddleocr`. Non-PDF files are decoded as
    UTF-8 text.
    """

    if filename.lower().endswith(".pdf"):
        if PdfReader is not None:
            try:
                reader = PdfReader(io.BytesIO(content))
                text = "".join(
                    page.extract_text() or "" for page in reader.pages
                ).strip()
                if text:
                    return text
            except Exception:
                pass

        try:
            from paddleocr import PaddleOCR  # imported lazily

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                ocr = PaddleOCR(use_angle_cls=True, lang="pt")
                result = ocr.ocr(tmp_path)
            finally:
                os.remove(tmp_path)

            text = "\n".join(
                " ".join(line[1][0] for line in page)
                for page in result
                if page
            ).strip()
            if text:
                return text
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Unable to extract text") from exc

        raise HTTPException(status_code=400, detail="Unable to extract text from PDF")

    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid file encoding") from exc


@app.post("/invoice-text")
async def upload_invoice_text(file: UploadFile = File(...)):
    """Upload an invoice and return only the extracted raw text."""
    content = await file.read()
    text = _extract_text(content, file.filename)
    return {"filename": file.filename, "text": text}


@app.post("/invoices")
async def upload_invoice(file: UploadFile = File(...)):
    """Upload an invoice and extract relevant fields using RAG."""
    content = await file.read()
    text = _extract_text(content, file.filename)

    vector_store.add_document(text)

    fields = identify_fields(text, FIELD_RULES, vector_store.chunks)
    headers = extract_product_headers(text, vector_store.chunks)
    fields["produtos"] = extract_products(text, headers, vector_store.chunks)
    return {"filename": file.filename, "fields": fields}
