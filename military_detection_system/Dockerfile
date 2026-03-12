FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Disable prompts during packages installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    git \
    libgl1-mesa-glx libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python3", "src/main.py"]
