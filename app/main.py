import io
import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Dict, List

from .agents import (
    identify_fields,
    select_relevant_chunks,
    extract_product_headers,
    extract_products,
)

# Optional image OCR helper. Imported lazily to avoid mandatory dependency on
# OpenCV/Tesseract during normal operation or testing environments where these
# heavy libraries might be absent.
try:  # pragma: no cover - availability depends on runtime environment
    from .ocr import image_to_text  # type: ignore
except Exception:  # pragma: no cover
    image_to_text = None  # type: ignore

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:  # pragma: no cover - handled via tests
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

app = FastAPI()


def _split_into_chunks(text: str, chunk_size: int = 200) -> List[str]:
    """Split ``text`` into chunks of up to ``chunk_size`` characters."""
    logger.debug("Splitting text of length %d into chunks of size %d", len(text), chunk_size)
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    logger.debug("Generated %d chunks", len(chunks))
    return chunks


class InMemoryVectorStore:
    """Very small in-memory vector store for demo purposes.

    The store keeps raw text chunks and uses ``select_relevant_chunks`` for
    similarity search. This mimics a vector database without external
    dependencies so tests remain lightweight.
    """

    def __init__(self) -> None:
        logger.debug("Initializing in-memory vector store")
        self.chunks: List[str] = []

    def add_document(self, text: str) -> None:
        logger.info("Adding document with %d characters to vector store", len(text))
        self.chunks.extend(_split_into_chunks(text))
        logger.debug("Vector store now holds %d chunks", len(self.chunks))

    def search(self, query: str, top_k: int = 3) -> List[str]:
        logger.info("Searching vector store for query '%s' (top_k=%d)", query, top_k)
        results = [chunk for _, chunk in select_relevant_chunks(query, self.chunks, top_k=top_k)]
        logger.debug("Search returned %d chunks", len(results))
        return results


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

    logger.info("Extracting text from file '%s'", filename)
    if filename.lower().endswith(".pdf"):
        logger.debug("File detected as PDF")
        if PdfReader is not None:
            try:
                reader = PdfReader(io.BytesIO(content))
                text = "".join(
                    page.extract_text() or "" for page in reader.pages
                ).strip()
                if text:
                    logger.debug("Text extracted via PyPDF2 (%d chars)", len(text))
                    return text
            except Exception as exc:
                logger.warning("PyPDF2 extraction failed: %s", exc)

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
                logger.debug("Text extracted via OCR (%d chars)", len(text))
                return text
        except Exception as exc:
            logger.error("OCR extraction failed: %s", exc)
            raise HTTPException(status_code=400, detail="Unable to extract text") from exc

        logger.error("No text extracted from PDF")
        raise HTTPException(status_code=400, detail="Unable to extract text from PDF")

    # Handle common image formats via Tesseract OCR if available
    if filename.lower().endswith((".png", ".jpg", ".jpeg")) and image_to_text:
        try:
            text = image_to_text(content)
            if text:
                logger.debug("Text extracted via image OCR (%d chars)", len(text))
                return text
        except Exception as exc:
            logger.error("Image OCR failed: %s", exc)
            raise HTTPException(status_code=400, detail="Unable to extract text from image") from exc

    try:
        text = content.decode("utf-8")
        logger.debug("Text extracted from non-PDF (%d chars)", len(text))
        return text
    except UnicodeDecodeError as exc:
        logger.error("Invalid file encoding: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid file encoding") from exc


@app.post("/invoice-text")
async def upload_invoice_text(file: UploadFile = File(...)):
    """Upload an invoice and return only the extracted raw text."""
    logger.info("/invoice-text called with file '%s'", file.filename)
    content = await file.read()
    logger.debug("Read %d bytes from uploaded file", len(content))
    text = _extract_text(content, file.filename)
    logger.info("Extracted %d characters of text", len(text))
    return {"filename": file.filename, "text": text}


@app.post("/invoices")
async def upload_invoice(file: UploadFile = File(...)):
    """Upload an invoice and extract relevant fields using RAG."""
    logger.info("/invoices called with file '%s'", file.filename)
    content = await file.read()
    logger.debug("Read %d bytes from uploaded file", len(content))
    text = _extract_text(content, file.filename)
    logger.info("Text extraction produced %d characters", len(text))

    vector_store.add_document(text)
    try:
        fields = identify_fields(text, FIELD_RULES, vector_store.chunks)
        logger.debug("Identified fields: %s", fields)
        headers = extract_product_headers(text, vector_store.chunks)
        logger.debug("Extracted headers: %s", headers)
        fields["produtos"] = extract_products(text, headers, vector_store.chunks)
        logger.debug("Extracted products: %s", fields["produtos"])
    except EnvironmentError as exc:
        logger.error("Environment configuration error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"filename": file.filename, "fields": fields}


@app.post("/invoices-gpt")
async def invoices_gpt(file: UploadFile = File(...)):
    """Upload an image of an invoice and extract products using OpenAI's APIs.
    The file is first uploaded to OpenAI's file API to obtain a ``file_id``.
    That ``file_id`` is then referenced in a call to the Responses API using
    model ``gpt-5`` with the prompt "gerar uma lista de produtos a partir dessa imagem".
    """
    logger.info("/invoices-gpt called with file '%s'", file.filename)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set")
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.error("openai package is not available: %s", exc)
        raise HTTPException(status_code=500, detail="openai package is not installed") from exc

    client = OpenAI(api_key=api_key)

    # Read file contents
    contents = await file.read()
    logger.debug("Read %d bytes from uploaded file", len(contents))

    try:
        uploaded = client.files.create(file=(file.filename, contents), purpose="vision")
        file_id = uploaded.id
        logger.info("File uploaded to OpenAI with id %s", file_id)

        response = client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "gerar uma lista de produtos a partir dessa imagem",
                        },
                        {
                            "type": "input_image",
                            "image": {"file_id": file_id},
                        },
                    ],
                }
            ],
        )
        result_text = getattr(response, "output_text", None)
        if result_text is None:
            try:
                # Fallback for different response formats
                result_text = response.output[0].content[0].text
            except Exception:
                result_text = str(response)
        logger.debug("Model response: %s", result_text)
    except Exception as exc:
        logger.error("OpenAI API call failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"file_id": file_id, "result": result_text}
