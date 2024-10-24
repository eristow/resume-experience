# TODO: reduce image size
# - remove unnecessary packages

# Stage 1: Build SQLite and Python
# TODO: upgrade to ubuntu22.04?
FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/local/lib:${LD_LIBRARY_PATH}

# Install build dependencies
RUN apt-get update && apt-get install -y \
	wget \
	gcc \
	g++ \
	make \
	zlib1g-dev \
	libssl-dev \
	libffi-dev \
	libbz2-dev \
	libreadline-dev \
	libncurses5-dev \
	libgdbm-dev \
	libnss3-dev \
	libncursesw5-dev \
	xz-utils \
	tk-dev \
	lzma \
	lzma-dev \
	liblzma-dev \
	--no-install-recommends \
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

# Build Python from source
ENV PYTHON_VERSION=3.12.2
RUN cd /tmp && \
	wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
	tar -xvf Python-${PYTHON_VERSION}.tgz && \
	cd Python-${PYTHON_VERSION} && \
	./configure \
	--enable-optimizations \
	--with-system-ffi \
	--with-computed-gotos \
	--enable-loadable-sqlite-extensions \
	--prefix=/usr/local && \
	make -j $(nproc) && \
	make altinstall && \
	cd .. && \
	rm -rf Python-${PYTHON_VERSION}* && \
	ldconfig


# Stage 2: Final image
FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:/usr/local/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/local/lib:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Copy built SQLite and Python from builder
COPY --from=builder /usr/local /usr/local

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
	libssl-dev \
	poppler-utils \
	tesseract-ocr \
	curl \
	--no-install-recommends \
	&& rm -rf /var/lib/apt/lists/*

# Create symbolic links
RUN ln -sf /usr/local/bin/python3.12 /usr/local/bin/python && \
	ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip

# Verify Python and SQLite3 versions
RUN python -c "import sqlite3; print('SQLite3 Version:', sqlite3.sqlite_version)"

WORKDIR /src

# Copy only necessary files
COPY requirements.txt .
COPY ./src .

# Install Python dependencies efficiently
RUN pip install --no-cache-dir -U pip setuptools wheel && \
	pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
	pip install --no-cache-dir \
	nvidia-cuda-runtime-cu12 \
	nvidia-cuda-nvrtc-cu12 \
	nvidia-cuda-cupti-cu12 \
	nvidia-cudnn-cu12 \
	bitsandbytes==0.41.3 && \
	pip install --no-cache-dir -r requirements.txt

# Create model directory
RUN mkdir -p models/mistral

# Run tests
RUN python -m pytest

# Cleanup unnecessary files
RUN apt-get clean && \
	rm -rf /var/lib/apt/lists/* && \
	find /usr/local/lib/python3.12 -type d -name "__pycache__" -exec rm -r {} + && \
	find /usr/local/lib/python3.12 -type d -name "tests" -exec rm -r {} + && \
	find /usr/local/lib/python3.12 -type d -name "test" -exec rm -r {} +

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Add CUDA verification script to run at container start
COPY ./streamlit-docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Volume for persisting models
VOLUME /models

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]