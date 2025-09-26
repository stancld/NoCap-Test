FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

LABEL name=nocap-test

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8

SHELL ["/bin/bash", "-c"]

ARG CUDA=12.8
ARG CUDNN=9.7.0.66-1

# CUDA NVCC & cuDNN Installation
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    cuda-nvcc-${CUDA/./-} \
    libcudnn9-cuda-12=${CUDNN} \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  # Prevent CUDA-related packages from being upgraded by apt-get upgrade later on
  && for pkg in cuda cuda-nvcc-${CUDA/./-} libcudnn9-cuda-12; do echo $pkg hold | dpkg --set-selections; done

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /bin/uv

ENV PYTHON=3.12
ENV PYTHON_DIR=/opt/python
# Use the virtual environment automatically
ENV PATH="$PYTHON_DIR/bin:$PATH"

RUN uv venv --python $PYTHON $PYTHON_DIR

ENV UV_TORCH_BACKEND=cu128

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN uv pip install -r requirements.txt
RUN uv pip install nvgpu

COPY . .

ARG RUNTIME=nvidia

CMD ["/bin/bash", "-c", ". /opt/python/bin/activate && exec bash"]
