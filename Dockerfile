FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
# Install Python 3.11.5 specifically
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support
RUN pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Install ptwt
RUN pip install ptwt==1.0.1

# Copy and install other requirements
COPY requirements.txt .
RUN grep -v -E "^(torch|torchvision|torchaudio|ptwt|nvidia-|triton)==" requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt gdown

# Download and setup data
RUN gdown --folder 1x2Hk4u_jrbbnJAIA5GDKXmuX456GMWQu -O data/
RUN if [ -d "data/MTS_Datasets" ]; then \
        mv data/MTS_Datasets/* data/ && \
        rm -rf data/MTS_Datasets; \
    fi

# Copy your code
COPY src/ ./src/
COPY figures/ ./figures/
COPY log/ ./log/
COPY checkpoint/ ./checkpoint/

COPY run_evaluations.sh .
RUN chmod +x run_evaluations.sh

RUN mkdir -p results

ENV PYTHONUNBUFFERED=1

CMD ["./run_evaluations.sh"]