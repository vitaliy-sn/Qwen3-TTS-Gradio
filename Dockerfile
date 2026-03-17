ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y build-essential python3 python3-pip python3-venv

ENV PYTHONUNBUFFERED=1
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
ENV VIRTUAL_ENV="/venv"

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ARG PIP_TRUSTED_HOST
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

ENV NVCC_THREADS="6"
ENV MAX_JOBS="6"
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV FLASH_ATTN_CUDA_ARCHS="120"
ENV FLASH_ATTENTION_FORCE_BUILD="TRUE"
ENV FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE"
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install flash-attn --no-build-isolation

RUN apt-get install -y libsndfile1 sox

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install gradio soundfile qwen-tts

WORKDIR /app

COPY app.py .

EXPOSE 7860

CMD ["python3", "/app/app.py"]
