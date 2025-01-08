# Stage 1: Build SQLite and Python
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 AS builder

LABEL maintainer="eristow"
LABEL version="1.0"
LABEL description="Streamlit Dockerfile with Mistral for resume-experience app"

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/local/lib:${LD_LIBRARY_PATH}

# Install build dependencies
RUN --mount=type=cache,target=/var/cache/apt \
	apt-get update && apt-get install -y \
	wget gcc g++ make \
	zlib1g-dev libssl-dev \
	&& rm -rf /var/lib/apt/lists/*

# Install newer SQLite3 from source
RUN cd /tmp && \
	wget https://www.sqlite.org/2024/sqlite-autoconf-3450000.tar.gz && \
	tar -xvf sqlite-autoconf-3450000.tar.gz && \
	cd sqlite-autoconf-3450000 && \
	./configure --prefix=/usr/local && \
	make && \
	make install && \
	cd .. && \
	rm -rf sqlite-autoconf-3450000* && \
	ldconfig

# Stage 2: Build Python dependencies
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 AS python-deps

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /deps

# Install Python and build dependencies
RUN apt-get update \
	&& DEBIAN_FRONTEND=noninteractive flock /var/lib/dpkg/lock-frontend -c ' \
	apt-get install -y --no-install-recommends \
	software-properties-common \
	' \
	&& add-apt-repository ppa:deadsnakes/ppa \
	&& DEBIAN_FRONTEND=noninteractive flock /var/lib/dpkg/lock-frontend -c ' \
	apt-get update \
	&& apt-get install -y --no-install-recommends \
	python3.12 python3.12-dev python3.12-venv \
	python3-venv python3-pip build-essential \
	' \
	&& rm -rf /var/lib/apt/lists/*

# Create and activate a Python 3.12 virtual environment
ENV VIRTUAL_ENV=/opt/venv
# RUN python3.12 -m venv /opt/venv
# RUN python3.12 -m venv /opt/venv
RUN python3.12 -v -m venv /opt/venv || (python3.12 -v -m venv --clear /opt/venv 2>&1 && false)
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python packages
COPY requirements_docker.txt ./requirements.txt
RUN grep -v 'torch==2.5.0' requirements.txt > requirements-no-torch.txt
RUN --mount=type=cache,target=/root/.cache/pip \
	pip install --no-cache-dir -U pip setuptools wheel && \
	pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
	pip install --no-cache-dir -r requirements-no-torch.txt

# Stage 3: Final image
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:/usr/local/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/local/lib:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_MODULE_LOADING=LAZY

# Copy built SQLite from builder
COPY --from=builder /usr/local/lib/libsqlite3* /usr/local/lib/
COPY --from=builder /usr/local/include/sqlite3*.h /usr/local/include/
COPY --from=python-deps /opt/venv /opt/venv

# Install Python and runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt \
	apt-get update && apt-get install -y \
	software-properties-common \
	&& add-apt-repository ppa:deadsnakes/ppa \
	&& apt-get update \
	&& apt-get install -y \
	python3.12 libpython3.12 curl \
	poppler-utils tesseract-ocr \
	&& rm -rf /var/lib/apt/lists/*

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"

# Verify the installation
RUN python -c "import torch; print('torch location:', torch.__file__)"

# Create model directory
RUN mkdir -p models/mistral

# Optimize networking stack
RUN echo 'net.ipv4.tcp_timestamps=1\n \
	net.ipv4.tcp_sack=1\n \
	net.core.rmem_max=16777216\n \
	net.core.wmem_max=16777216\n \
	net.ipv4.tcp_rmem=4096 87380 16777216\n \
	net.ipv4.tcp_wmem=4096 65536 16777216\n \
	net.ipv4.tcp_window_scaling=1' >> /etc/sysctl.conf

WORKDIR /src

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

COPY ./streamlit-docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

VOLUME /models

# Verify CUDA is available
# RUN python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA is available')"

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]