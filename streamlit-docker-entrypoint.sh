#!/bin/bash
set -e

check_model_files() {
    local model_dir="$1"
    local missing_files=()
    local required_files=(
        "config.json"
        "generation_config.json"
        "tokenizer.json"
        "tokenizer_config.json"
        "special_tokens_map.json"
    )
    
    # Check if directory exists
    if [ ! -d "$model_dir" ]; then
        echo "Model directory not found"
        echo "MISSING_FILES=directory"
        return 1
    fi
    
    # Check for required files
    for file in "${required_files[@]}"; do
        if [ ! -f "$model_dir/$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    # Check for model file(s)
    local model_found=false
    local total_shards=0
    local missing_shards=()

    # Check for non-sharded model
    if [ -f "$model_dir/model.safetensors" ]; then
        echo "Found non-sharded model file"
        model_found=true
    else
        # Look for any shard to determine total count
        for shard in "$model_dir"/model-*-of-*.safetensors; do
            if [ -f "$shard" ]; then
                total_shards=$(echo "$shard" | grep -o 'of-[0-9]*' | cut -d'-' -f2)
                model_found=true
                break
            fi
        done

        if [ "$model_found" = true ]; then
            echo "Found sharded model. Checking all $total_shards shards..."
            local total_shards_num=$(echo "$total_shards" | grep -o '[0-9]*')
            echo "Total shards: $total_shards_num"

            # Check if all shards exist
            for ((i=1; i<=$total_shards_num; i++)); do
                local shard_name=$(printf "model-%05d-of-%05s.safetensors" $i $total_shards)
                if [ ! -f "$model_dir/$shard_name" ]; then
                    missing_shards+=("$shard_name")
                fi
            done
        fi
    fi

    if [ "$model_found" = false ]; then
        missing_files+=("model.safetensors")
        echo "No model files found"
        echo "MISSING_FILES=${missing_files[*]}"
        return 1
    fi

    # If we have missing files or shards, return failure
    if [ ${#missing_files[@]} -ne 0 ] || [ ${#missing_shards[@]} -ne 0 ]; then
        if [ ${#missing_files[@]} -ne 0 ]; then
            echo "Missing required files: ${missing_files[*]}"
        fi
        if [ ${#missing_shards[@]} -ne 0 ]; then
            echo "Missing shards: ${missing_shards[*]}"
        fi
        # Export all missing files for the Python script
        echo "MISSING_FILES=${missing_files[*]} ${missing_shards[*]}"
        return 1
    fi

    echo "All required files and model shards found"
    echo "MISSING_FILES="
    return 0
}


# Output directory structure
ls -la /models
ls -la /src/models
# ls -la /src/models/mistral

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
MODEL_DIR="/models/mistral"
if ! check_model_files "$MODEL_DIR"; then
    echo "Model files not found or incomplete in volume, downloading..."
    export MISSING_FILES
    python scripts/docker_download_model.py
else
    echo "All required model files found in volume, skipping download."
    echo "Found files:"
    ls -la "$MODEL_DIR"
fi

# Start Streamlit
exec python -m streamlit run main.py --server.port=8501 --server.address=0.0.0.0
