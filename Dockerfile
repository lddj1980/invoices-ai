FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn PyPDF2 paddleocr openai python-multipart fpdf

# Copy project files
COPY . .

# Expose application port
EXPOSE 8000

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
