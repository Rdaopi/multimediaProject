FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /workspace/app

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .

RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

RUN pip install --default-timeout=1000 --no-cache-dir \
    av ctranslate2 cython dtw-python einx einops-exts \
    faster-whisper ffmpy ftfy langid language-tags \
    openai openai-whisper pandas pydub resampy \
    sentencepiece vector-quantize-pytorch wavmark whisper-timestamped \
    colorama coloredlogs

COPY . .

ENTRYPOINT ["python", "main.py"]