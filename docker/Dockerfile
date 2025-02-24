# tms_efield_prediction/docker/Dockerfile
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy setup script
COPY setup.py .
COPY README.md .

# Copy source code
COPY tms_efield_prediction/ tms_efield_prediction/

# Install the package in development mode
RUN pip install -e .

# Create directories for data and experiments
RUN mkdir -p /data/raw \
    /data/processed/dual_modal \
    /data/processed/oriented_mri \
    /data/processed/mri_efield \
    /experiments/dual_modal \
    /experiments/oriented_mri \
    /experiments/mri_efield

# Set environment variables
ENV PYTHONPATH=/app
ENV DATA_DIR=/data
ENV EXPERIMENT_DIR=/experiments

# Default command
CMD ["bash"]