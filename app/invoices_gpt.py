import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

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
                            "image_file": {"file_id": file_id},
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
