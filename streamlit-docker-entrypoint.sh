#!/bin/bash
set -e

# Create symlink from mounted volume to application directory if needed
if [ ! -L "/src/models" ] && [ -d "/models" ]; then
    echo "Creating symlink for models directory..."
    rm -rf /src/models
    ln -s /models /src/models
fi

# Verify CUDA setup
echo "Verifying CUDA setup..."
python -c "import torch; print('PyTorch version:', torch.__version__); \
    print('CUDA available:', torch.cuda.is_available()); \
    print('CUDA version:', torch.version.cuda); \
    print('Device count:', torch.cuda.device_count()); \
    print('Current device:', torch.cuda.current_device() if torch.cuda.is_available() else 'None')"

# Run CUDA verification script
python scripts/cuda_verify.py

# Download model if not already present
if [ -z "$( ls -A '/models/mistral/' )" ]; then
    echo "Model not found in volume, downloading..."
    python scripts/docker_download_model.py
else
    echo "Model found in volume, skipping download."
fi

# Start Streamlit
exec python -m streamlit run main.py --server.port=8501 --server.address=0.0.0.0
