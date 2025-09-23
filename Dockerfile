# ---- Base image ----
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/SlowFastTrainer

# Minimal system deps
RUN apt-get update && apt-get install -y \
    python3-pip python3-venv git curl ffmpeg unzip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Bring code + requirements (we install at runtime, not during build)
COPY requirements.txt .
COPY train_slowfast.py config.py get_RTSP_stream.py data_formatting.py ./

# ---- Bootstrap-on-first-run entrypoint script ----
RUN cat >/usr/local/bin/bootstrap-and-run.sh <<'EOF'
#!/usr/bin/env bash
set -e
VENV=/opt/venv
mkdir -p "$VENV/tmp"

if [ ! -e "$VENV/bin/activate" ]; then
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  python -m pip install --upgrade pip

  # Install GPU torch first (put temp files in venv tmp on the mounted volume)
  TMPDIR="$VENV/tmp" pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1

  # Fix problematic pins; prefer headless OpenCV to save space
  sed -i '/^c-ares==/d' requirements.txt || true
  sed -i 's/^cairo==.*/pycairo>=1.26,<1.27/' requirements.txt || true
  if grep -q '^opencv-python==' requirements.txt; then
    sed -i 's/^opencv-python==/opencv-python-headless==/' requirements.txt
  fi

  # Rest of the deps
  TMPDIR="$VENV/tmp" pip install --no-cache-dir --prefer-binary -r requirements.txt
fi

exec "$VENV/bin/python" train_slowfast.py "$@"
EOF

RUN chmod +x /usr/local/bin/bootstrap-and-run.sh
ENTRYPOINT ["/usr/local/bin/bootstrap-and-run.sh"]
