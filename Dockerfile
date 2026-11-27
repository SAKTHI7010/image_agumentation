FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# Non-root user required by Hugging Face Spaces (uid 1000)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# System libraries needed by OpenCV headless & pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=user:user . .

RUN mkdir -p /home/user/app/outputs && chown -R user:user /home/user/app/outputs
USER user

EXPOSE 10000

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
