FROM python:3.10-bullseye

WORKDIR /app

# Install system dependencies including those needed for llama-cpp-python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for llama-cpp-python build
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=OFF"
ENV FORCE_CMAKE=1

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p posts_cache vector_store logs

# Copy application code
COPY . .

# Default command
CMD ["python", "src/app.py"]
