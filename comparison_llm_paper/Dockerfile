FROM python:3.13-slim

# Install ffmpeg (required by Whisper for audio loading)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Prevent native crashes (exit 139 / SIGSEGV) from OpenMP runtime conflicts
# between TensorFlow (YAMNet) and PyTorch (Whisper/WavLM) loaded in one process.
ENV KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1

# Dump full amplitude + spectrum arrays as .npz artifacts for every
# processed chunk (offline analysis). Baked into the image so no CDK task
# definition change is needed. Set to "false" to keep only summary stats.
ENV DUMP_AMPLITUDE_ARRAYS=true

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps nussl>=1.1.9

# Copy application code
COPY src/ src/

# Entry point — overridden by ECS task environment variables
CMD ["python", "-m", "src.pipelines.agentic_pipeline"]
