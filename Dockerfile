FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only the packages needed for training (no Jupyter etc.)
RUN pip3 install --no-cache-dir \
    torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install --no-cache-dir \
    transformers==5.8.1 \
    accelerate==1.13.0 \
    peft==0.13.2 \
    safetensors==0.7.0 \
    tokenizers==0.22.2 \
    tqdm==4.67.3 \
    datasets==4.8.5 \
    huggingface_hub==1.15.0 \
    mo-sql-parsing \
    python-dotenv

# Copy project files (train_full.py now lives under src/full_finetuning/, and
# imports other modules via "src...." package paths)
COPY pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/

# Editable install so the "src...." imports work regardless of the working
# directory (same mechanism used for local development, see README)
RUN pip3 install --no-cache-dir --no-deps -e .

CMD ["python3", "src/full_finetuning/train_full.py"]
