import io
import os
import tempfile
import logging
import uuid
import time
import json
import urllib.error
import urllib.request
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from typing import Dict, List

from .prompts import (
    INVOICE_PARSER_STEP1_PROMPT,
    INVOICE_PARSER_STEP2_PROMPT,
)

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

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)

try:  # pragma: no cover - handled via tests
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

app = FastAPI()

# Simple in-memory job store for background processing results
jobs: dict[str, dict] = {}
JOBS_TTL_SECONDS = 3600
WEBHOOK_URL = "https://pemmlybnqjvpxmgzxqhl.supabase.co/functions/v1/webhook-invoice-result"


def _cleanup_jobs() -> None:
    """Remove jobs older than ``JOBS_TTL_SECONDS`` to prevent unbounded growth."""
    now = time.time()
    logger.debug("Running job cleanup at %.3f with %d jobs tracked", now, len(jobs))
    expired = [
        jid
        for jid, info in jobs.items()
        if now - info.get("created_at", now) > JOBS_TTL_SECONDS
    ]
    if expired:
        logger.info("Removing %d expired jobs: %s", len(expired), expired)
    for jid in expired:
        jobs.pop(jid, None)
    logger.debug("Cleanup complete. Remaining jobs: %d", len(jobs))


def _post_webhook(payload: dict) -> None:
    """Send ``payload`` to the external webhook endpoint.

    Errors are logged but do not raise exceptions so the background processing
    flow is not interrupted.
    """

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        WEBHOOK_URL, data=data, headers={"Content-Type": "application/json"}
    )
    logger.debug(
        "Dispatching webhook to %s with payload keys: %s",
        WEBHOOK_URL,
        list(payload),
    )
    try:
        with urllib.request.urlopen(request, timeout=10):  # nosec B310 - trusted URL
            logger.info(
                "Webhook dispatched successfully for external_job_id=%s",
                payload.get("external_job_id"),
            )
    except urllib.error.URLError as exc:
        logger.error("Failed to dispatch webhook: %s", exc)


def _notify_webhook(job_id: str) -> None:
    """Dispatch webhook notifications when a job finishes."""

    logger.debug("Notifying webhook for job_id=%s", job_id)
    job = jobs.get(job_id)
    if not job:
        logger.warning("Job %s not found during webhook notification", job_id)
        return

    external_job_id = job.get("external_job_id")
    if not external_job_id:
        return

    status = job.get("status")
    logger.debug("Job %s current status: %s", job_id, status)
    if status == "done":
        payload = {
            "external_job_id": external_job_id,
            "status": "completed",
            "result": job.get("result"),
        }
    elif status == "error":
        error_message = job.get("result")
        if error_message is None:
            error_message = "Unknown error"
        payload = {
            "external_job_id": external_job_id,
            "status": "error",
            "error_message": error_message if isinstance(error_message, str) else str(error_message),
        }
    else:
        return

    _post_webhook(payload)


def _finalize_job(job_id: str, status: str, result) -> None:
    """Update job status/result and emit webhook notifications when required."""

    logger.debug("Finalizing job %s with status %s", job_id, status)
    job = jobs.get(job_id)
    if job is None:
        logger.warning("Attempted to finalize unknown job %s", job_id)
        return

    job["status"] = status
    job["result"] = result
    logger.info(
        "Job %s finalized. Status=%s | Result type=%s",
        job_id,
        status,
        type(result).__name__,
    )
    _notify_webhook(job_id)


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
        logger.info(
            "Adding document with %d characters to vector store (current_chunks=%d)",
            len(text),
            len(self.chunks),
        )
        self.chunks.extend(_split_into_chunks(text))
        logger.debug("Vector store now holds %d chunks", len(self.chunks))

    def search(self, query: str, top_k: int = 3) -> List[str]:
        logger.info(
            "Searching vector store for query '%s' (top_k=%d | total_chunks=%d)",
            query,
            top_k,
            len(self.chunks),
        )
        results = [
            chunk for _, chunk in select_relevant_chunks(query, self.chunks, top_k=top_k)
        ]
        logger.debug(
            "Search returned %d chunks. First chunk preview: %s",
            len(results),
            (results[0][:80] + "...") if results else None,
        )
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

DEFAULT_PDF_PROMPT = (
    "gerar uma lista de produtos a partir deste PDF. "
    "Se possível, retorne em JSON com campos como "
    "descricao, quantidade, unidade, preco_unitario, moeda e total."
)

DEFAULT_PDF_FALLBACK_PROMPT_PREFIX = (
    "gerar uma lista de produtos a partir do conteúdo a seguir (extraído de um PDF). "
    "Se possível, retorne em JSON com campos como descricao, quantidade, unidade, "
    "preco_unitario, moeda e total. Conteúdo:\n\n"
)

def _json_to_pdf(data: Dict) -> bytes:
    """Convert a JSON-serializable object to PDF bytes."""
    logger.info("Converting JSON payload to PDF (keys=%s)", list(data)[:10])
    try:  # pragma: no cover - availability depends on runtime
        from fpdf import FPDF
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.error("fpdf package is not installed: %s", exc)
        raise HTTPException(status_code=500, detail="fpdf package is not installed") from exc

    pdf = FPDF()
    pdf.add_page()

    font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
    if not os.path.exists(font_path):
        logger.error("Unicode font file not found at %s", font_path)
        raise HTTPException(status_code=500, detail="Unicode font file not found")

    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    logger.debug("JSON converted to string with %d characters", len(json_str))
    for line in json_str.splitlines():
        pdf.multi_cell(0, 10, line)
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    logger.info("Generated PDF with %d bytes from JSON", len(pdf_bytes))
    return pdf_bytes

def _extract_text(content: bytes, filename: str) -> str:
    """Extract text from ``content`` using PDF parsing or OCR.

    For PDFs the function first attempts :class:`PyPDF2.PdfReader`. When no text
    is found it falls back to :mod:`paddleocr`. Non-PDF files are decoded as
    UTF-8 text.
    """

    logger.info("Extracting text from file '%s' (%d bytes)", filename, len(content))
    if filename.lower().endswith(".pdf"):
        logger.debug("File detected as PDF")
        if PdfReader is not None:
            try:
                reader = PdfReader(io.BytesIO(content))
                page_count = len(reader.pages)
                logger.debug("PdfReader initialized with %d pages", page_count)
                text = "".join(page.extract_text() or "" for page in reader.pages).strip()
                if text:
                    logger.info(
                        "Text extracted via PyPDF2 (%d chars | pages=%d)",
                        len(text),
                        page_count,
                    )
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
                logger.debug("Running PaddleOCR on temporary file %s", tmp_path)
                result = ocr.ocr(tmp_path)
                logger.debug("PaddleOCR returned %d pages", len(result) if result else 0)
            finally:
                os.remove(tmp_path)

            text = "\n".join(
                " ".join(line[1][0] for line in page)
                for page in result
                if page
            ).strip()
            if text:
                logger.info("Text extracted via OCR (%d chars)", len(text))
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
                logger.info("Text extracted via image OCR (%d chars)", len(text))
                return text
        except Exception as exc:
            logger.error("Image OCR failed: %s", exc)
            raise HTTPException(status_code=400, detail="Unable to extract text from image") from exc

    try:
        text = content.decode("utf-8")
        logger.info("Text extracted from non-PDF (%d chars)", len(text))
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
    logger.debug("Returning text payload preview: %s", text[:200])
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
        logger.debug("Starting field identification using %d stored chunks", len(vector_store.chunks))
        fields = identify_fields(text, FIELD_RULES, vector_store.chunks)
        logger.debug("Identified fields: %s", fields)
        headers = extract_product_headers(text, vector_store.chunks)
        logger.debug("Extracted headers: %s", headers)
        fields["produtos"] = extract_products(text, headers, vector_store.chunks)
        logger.debug("Extracted products: %s", fields["produtos"])
    except EnvironmentError as exc:
        logger.error("Environment configuration error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("Invoice processing completed successfully for '%s'", file.filename)
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

    client = OpenAI(api_key=api_key, timeout=20 * 60)

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

    logger.info("/invoices-gpt completed successfully for file '%s'", file.filename)
    return {"file_id": file_id, "result": result_text}


@app.post("/invoice-json-gpt-pdf")
async def invoice_json_gpt_pdf(
    background_tasks: BackgroundTasks,
    packing_list: UploadFile = File(...),
    invoice: str = Form(...),
    prompt: str | None = Form(None),
    external_job_id: str | None = Form(None),
):
    """Recebe um JSON de invoice e um PDF de packing list.

    O JSON é convertido para PDF e então processado junto com o PDF enviado,
    reutilizando :func:`_process_invoices_gpt_pdf`.
    """
    logger.info(
        "/invoice-json-gpt-pdf called with packing_list '%s'", packing_list.filename
    )
    if prompt:
        logger.debug(
            "Custom prompt provided (%d chars) for invoice_json_gpt_pdf",
            len(prompt),
        )
    else:
        logger.debug("No custom prompt provided for invoice_json_gpt_pdf")

    if not packing_list.filename.lower().endswith(".pdf"):
        logger.error("File is not a PDF: %s", packing_list.filename)
        raise HTTPException(status_code=400, detail="Envie um arquivo PDF.")

    try:
        invoice_data = json.loads(invoice)
    except Exception:
        logger.error("Invalid JSON provided for invoice")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        invoice_contents = _json_to_pdf(invoice_data)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to generate PDF from JSON: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate invoice PDF") from exc

    packing_contents = await packing_list.read()
    logger.debug(
        "Read %d bytes for packing_list and generated %d bytes for invoice PDF",
        len(packing_contents),
        len(invoice_contents),
    )

    job_id = str(uuid.uuid4())
    logger.info(
        "Scheduling invoice_json_gpt_pdf job_id=%s (external_job_id=%s)",
        job_id,
        external_job_id,
    )
    jobs[job_id] = {
        "status": "pending",
        "result": None,
        "created_at": time.time(),
        "external_job_id": external_job_id,
    }
    _cleanup_jobs()
    background_tasks.add_task(
        _process_invoices_gpt_pdf, job_id, packing_contents, invoice_contents, prompt
    )
    logger.debug("Background task registered for job_id=%s", job_id)
    return {"job_id": job_id}


@app.post("/invoices-gpt-pdf")
async def invoices_gpt_pdf(
    background_tasks: BackgroundTasks,
    packing_list: UploadFile = File(...),
    invoice: UploadFile = File(...),
    prompt: str | None = Form(None),
    external_job_id: str | None = Form(None),
):
    """Agenda o processamento de dois PDFs e retorna um ``job_id`` para consulta."""
    logger.info(
        "/invoices-gpt-pdf called with packing_list '%s' and invoice '%s'",
        packing_list.filename,
        invoice.filename,
    )
    if prompt:
        logger.debug(
            "Custom prompt provided (%d chars) for invoices_gpt_pdf",
            len(prompt),
        )
    else:
        logger.debug("No custom prompt provided for invoices_gpt_pdf")

    for f in (packing_list, invoice):
        if not f.filename.lower().endswith(".pdf"):
            logger.error("File is not a PDF: %s", f.filename)
            raise HTTPException(status_code=400, detail="Envie dois arquivos PDF.")

    packing_contents = await packing_list.read()
    invoice_contents = await invoice.read()
    logger.debug(
        "Read %d bytes for packing_list and %d bytes for invoice",
        len(packing_contents),
        len(invoice_contents),
    )

    job_id = str(uuid.uuid4())
    logger.info(
        "Scheduling invoices_gpt_pdf job_id=%s (external_job_id=%s)",
        job_id,
        external_job_id,
    )
    jobs[job_id] = {
        "status": "pending",
        "result": None,
        "created_at": time.time(),
        "external_job_id": external_job_id,
    }
    _cleanup_jobs()
    background_tasks.add_task(
        _process_invoices_gpt_pdf, job_id, packing_contents, invoice_contents, prompt
    )
    logger.debug("Background task registered for job_id=%s", job_id)
    return {"job_id": job_id}


def _process_invoices_gpt_pdf(
    job_id: str,
    packing_contents: bytes,
    invoice_contents: bytes,
    prompt: str | None,
) -> None:
    """Executa o processamento pesado e armazena o resultado em ``jobs``."""
    logger.info(
        "Starting _process_invoices_gpt_pdf for job_id=%s | packing_bytes=%d | invoice_bytes=%d | prompt=%s",
        job_id,
        len(packing_contents),
        len(invoice_contents),
        "custom" if prompt else "default",
    )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        _finalize_job(job_id, "error", "OPENAI_API_KEY is not set")
        return

    try:
        from openai import OpenAI
    except Exception:
        _finalize_job(job_id, "error", "openai package is not installed")
        return

    client = OpenAI(api_key=api_key, timeout=20 * 60)
    custom_prompt = prompt is not None
    if prompt is None:
        prompt = DEFAULT_PDF_PROMPT
        logger.debug("Using default PDF prompt")
    else:
        logger.debug("Using custom prompt with %d characters", len(prompt))

    uploaded_ids: dict[str, str] = {}
    result_text = ""
    try:
        upl_pack = client.files.create(
            file=("packing_list.pdf", packing_contents), purpose="assistants"
        )
        uploaded_ids["packing_list"] = upl_pack.id
        upl_inv = client.files.create(
            file=("invoice.pdf", invoice_contents), purpose="assistants"
        )
        uploaded_ids["invoice"] = upl_inv.id
        logger.info("Uploaded files to OpenAI with ids %s", uploaded_ids)

        response = client.responses.create(
            model="gpt-5",
            input=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": uploaded_ids["invoice"]},
                        {"type": "input_file", "file_id": uploaded_ids["packing_list"]},
                    ],
                },
            ],
        )

        result_text = getattr(response, "output_text", None)
        if result_text is None:
            try:
                result_text = response.output[0].content[0].text
            except Exception:
                result_text = str(response)
        logger.debug(
            "OpenAI response received for job_id=%s (custom_prompt=%s)",
            job_id,
            custom_prompt,
        )

    except Exception as exc:
        logger.error("OpenAI API call (PDF) failed: %s", exc)
        try:
            logger.info("Falling back to local text extraction for job_id=%s", job_id)
            pack_text = _extract_text(packing_contents, "packing_list.pdf")
            inv_text = _extract_text(invoice_contents, "invoice.pdf")
            combined = f"Packing List:\n{pack_text}\n\nInvoice:\n{inv_text}"
            logger.info(
                "Local text extraction succeeded with %d chars; sending as input_text fallback",
                len(combined),
            )
            response = client.responses.create(
                model="gpt-5",
                input=[
                    {
                        "role": "system",
                        "content": prompt if custom_prompt else DEFAULT_PDF_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": combined}
                        ],
                    },
                ],
            )
            result_text = getattr(response, "output_text", None) or str(response)
            uploaded_ids = {}
            logger.info("Fallback OpenAI text-only call succeeded for job_id=%s", job_id)
        except Exception as exc_inner:
            logger.error("Fallback local extraction also failed: %s", exc_inner)
            _finalize_job(job_id, "error", str(exc_inner))
            return
    finally:
        for fid in uploaded_ids.values():
            try:
                client.files.delete(fid)
            except Exception as exc_del:
                logger.warning("Failed to delete OpenAI file %s: %s", fid, exc_del)

    _finalize_job(
        job_id,
        "done",
        {"file_ids": uploaded_ids or None, "result": result_text},
    )
    logger.info("_process_invoices_gpt_pdf completed successfully for job_id=%s", job_id)


def _process_invoice_parser(
    job_id: str, packing_contents: bytes, invoice_contents: bytes
) -> None:
    """Orchestrate two-step invoice parsing using fixed prompts.

    Step 1 uses :data:`INVOICE_PARSER_STEP1_PROMPT` to extract a JSON array
    of goods from the provided invoice and packing list PDFs. The resulting
    JSON is converted to PDF and fed into Step 2 with
    :data:`INVOICE_PARSER_STEP2_PROMPT` to complete the weight fields.
    """

    logger.info(
        "Starting _process_invoice_parser for job_id=%s | packing_bytes=%d | invoice_bytes=%d",
        job_id,
        len(packing_contents),
        len(invoice_contents),
    )
    stage1_job = f"{job_id}_stage1"
    jobs[stage1_job] = {"status": "pending", "result": None, "created_at": time.time()}
    _process_invoices_gpt_pdf(
        stage1_job, packing_contents, invoice_contents, INVOICE_PARSER_STEP1_PROMPT
    )
    stage1 = jobs.pop(stage1_job, None)
    if not stage1 or stage1.get("status") != "done":
        logger.error("Stage 1 failed for job_id=%s with result=%s", job_id, stage1)
        _finalize_job(job_id, "error", stage1.get("result") if stage1 else None)
        return

    result_text = stage1.get("result")
    if isinstance(result_text, dict):
        result_text = result_text.get("result", "")
    try:
        invoice_json = json.loads(result_text)
    except Exception:
        logger.error("Invalid JSON output from stage 1 for job_id=%s", job_id)
        _finalize_job(job_id, "error", "Invalid JSON from stage 1")
        return

    try:
        invoice_pdf = _json_to_pdf(invoice_json)
    except HTTPException as exc:
        logger.error("Failed to convert stage 1 JSON to PDF for job_id=%s: %s", job_id, exc.detail)
        _finalize_job(job_id, "error", exc.detail)
        return
    except Exception as exc:  # pragma: no cover
        logger.error("Unexpected error creating PDF for job_id=%s: %s", job_id, exc)
        _finalize_job(job_id, "error", str(exc))
        return

    logger.info("Stage 1 completed successfully for job_id=%s, launching stage 2", job_id)
    _process_invoices_gpt_pdf(
        job_id, packing_contents, invoice_pdf, INVOICE_PARSER_STEP2_PROMPT
    )
    logger.info("_process_invoice_parser finished for job_id=%s", job_id)


@app.post("/invoice-parser")
async def invoice_parser(
    background_tasks: BackgroundTasks,
    packing_list: UploadFile = File(...),
    invoice: UploadFile = File(...),
    external_job_id: str | None = Form(None),
):
    """Execute the two-step invoice parsing pipeline.

    Both ``packing_list`` and ``invoice`` must be PDFs. The endpoint schedules
    a background task and returns a ``job_id`` for status polling.
    """

    for f in (packing_list, invoice):
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Envie dois arquivos PDF.")

    packing_contents = await packing_list.read()
    invoice_contents = await invoice.read()
    logger.debug(
        "Read %d bytes for packing_list and %d bytes for invoice in invoice_parser",
        len(packing_contents),
        len(invoice_contents),
    )

    job_id = str(uuid.uuid4())
    logger.info(
        "Scheduling invoice_parser job_id=%s (external_job_id=%s)",
        job_id,
        external_job_id,
    )
    jobs[job_id] = {
        "status": "pending",
        "result": None,
        "created_at": time.time(),
        "external_job_id": external_job_id,
    }
    _cleanup_jobs()
    background_tasks.add_task(
        _process_invoice_parser, job_id, packing_contents, invoice_contents
    )
    logger.debug("Background task registered for invoice_parser job_id=%s", job_id)
    return {"job_id": job_id}


@app.get("/invoice-parser/{job_id}")
async def get_invoice_parser(job_id: str):
    """Return status/result for the ``invoice-parser`` pipeline."""
    _cleanup_jobs()
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    logger.debug("Returning status for invoice_parser job_id=%s: %s", job_id, job["status"])
    return {"status": job["status"], "result": job["result"]}


@app.get("/invoices-gpt-pdf/{job_id}")
async def get_invoices_gpt_pdf(job_id: str):
    """Retorna o status e o resultado de um job."""
    _cleanup_jobs()
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    logger.debug("Returning status for invoices_gpt_pdf job_id=%s: %s", job_id, job["status"])
    return {"status": job["status"], "result": job["result"]}


