# Use an official RunPod PyTorch image matching your requirements
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-ubuntu22.04 

WORKDIR /app

# Set environment variables if needed (e.g., for Hugging Face cache)
# ENV TRANSFORMERS_CACHE=/app/cache/transformers
# ENV HF_HOME=/app/cache/huggingface
ENV PYTHONUNBUFFERED=1

# Install essential OS packages and cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Detectron2 (check official guide for specific PyTorch/CUDA versions)
# This command is an example, VERIFY it for torch 2.0.1 / cuda 11.8
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# Or install a specific commit/version if needed:
# RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6' # Example v0.6

# Copy dependency file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code into the container
COPY . .

# Download models during build (optional, but good for consistency)
# If models are large, use RunPod network volumes instead.
# This requires models to be accessible (e.g., download script or copied)
# RUN python -c "from tryon_logic import initialize_models; initialize_models()"
# OR download specific files:
# RUN wget -P ./ckpt/densepose https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl
# Make sure BASE_PATH in tryon_logic.py points to where models are downloaded/stored.
# If using HF models, they'll download on first run if not built-in.

# Expose the port FastAPI will run on (matches uvicorn command)
EXPOSE 8000

# Command to run the application (will be overridden by RunPod handler)
# This is mainly for local testing of the container image
# CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "8000"]