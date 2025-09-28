FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && pip3 install -r requirements.txt

COPY . /app

EXPOSE 80
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4"]
