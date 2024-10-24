import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from model_token import ACCESS_TOKEN


def check_system_resources():
    """Check and print system resources"""
    import psutil

    memory = psutil.virtual_memory()
    print(f"\nSystem Memory Status:")
    print(f"Total: {memory.total / (1024**3):.2f} GB")
    print(f"Available: {memory.available / (1024**3):.2f} GB")
    print(f"Used: {memory.used / (1024**3):.2f} GB")
    print(f"\nGPU Status:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(
            f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        )


def download_and_save_model():
    """Download and save the quantized model"""
    BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    save_directory = "/models/mistral"
    os.makedirs(save_directory, exist_ok=True)

    check_system_resources()
    print("\nInitializing download process...")

    try:
        # Initialize tokenizer first
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=ACCESS_TOKEN)

        print("Saving tokenizer...")
        tokenizer.save_pretrained(save_directory)

        # Configure quantization
        print("Configuring 4-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            token=ACCESS_TOKEN,
            device_map="auto",
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
        )

        print("Saving model...")
        # Save in float32 format since we can't save quantized model directly
        model = model.to(torch.float32)
        model.save_pretrained(
            save_directory,
            max_shard_size="1GB",
            safe_serialization=True,
        )

        print(
            f"Model and tokenizer saved successfully to: {os.path.abspath(save_directory)}"
        )
        print(
            "\nNote: Model has been saved in full precision format. It will be automatically quantized when loaded for inference"
        )

    except Exception as e:
        print(f"\nError during model processing: {str(e)}")
        print("\nFull error details:")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires GPU support.")
        sys.exit(1)

    download_and_save_model()
