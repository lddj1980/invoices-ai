from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Dict, List

from .agents import identify_field, select_relevant_chunks

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
        return select_relevant_chunks(query, self.chunks, top_k=top_k)


vector_store = InMemoryVectorStore()

FIELD_RULES: Dict[str, str] = {
    "sales_condition": "Identify the sales condition code present in the invoice.",
}


@app.post("/invoices")
async def upload_invoice(file: UploadFile = File(...)):
    """Upload an invoice and extract relevant fields using RAG."""

    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid file encoding") from exc

    vector_store.add_document(text)

    fields = {}
    for field_name, rules in FIELD_RULES.items():
        fields[field_name] = identify_field(
            field_name, text, rules, vector_store.chunks
        )
    return {"filename": file.filename, "fields": fields}
