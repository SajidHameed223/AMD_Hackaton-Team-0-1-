# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

# ---------- dependencies ----------
COPY requirements.txt .

# Install CPU-only PyTorch first (saves ~1.5 GB vs CUDA variant),
# then the remaining deps. --no-cache-dir keeps the layer small.
RUN pip install --no-cache-dir \
        torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ---------- application code ----------
COPY app/  app/
COPY local/ local/
COPY ml/ ml/
COPY solve.py .

# ---------- local model weights (optional) ----------
# Pre-download weights with:  python download_model.py
# If models/ is empty the image stays small (~1.5 GB) and T1 fails
# gracefully at runtime, falling back to T2 (cloud).
COPY models/ models/

# Point local/model.py at the bundled weights (if present)
ENV MODEL_NAME=/app/models/gemma-2b-it

# ---------- runtime ----------
# Grading harness injects FIREWORKS_API_KEY, FIREWORKS_BASE_URL,
# and ALLOWED_MODELS at runtime — solve.py reads them from env.
ENTRYPOINT ["python", "solve.py"]
