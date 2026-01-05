FROM python:3.11-slim

# Install system deps for LibreOffice (DOCX->PDF) and WeasyPrint (cairo/pango)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    fonts-liberation \
    libglib2.0-0 libpango-1.0-0 libgdk-pixbuf-2.0-0 libffi-dev libcairo2 libpangoft2-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose envs (OpenAI/LangSmith/CORS)
ENV OPENAI_API_KEY=changeme \
    AI_CORS_ALLOW_ORIGINS=* \
    LANGCHAIN_TRACING_V2=false \
    LANGCHAIN_ENDPOINT=https://api.smith.langchain.com \
    LANGCHAIN_PROJECT=agent-btp

EXPOSE 8000

CMD ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
