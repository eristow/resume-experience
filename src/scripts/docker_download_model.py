import os
import sys
import torch
import logging
import shutil
import psutil
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as transformers_logging
from model_token import ACCESS_TOKEN


def setup_logging():
    """Configure logging for detailed output"""
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Set transformers logging to INFO
    transformers_logging.set_verbosity_info()

    # Enable tensorboard logging in transformers
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"

    # Create logger for this script
    logger = logging.getLogger("model_download")
    logger.setLevel(logging.INFO)

    return logger


def check_system_resources(logger):
    """Check and print system resources"""
    memory = psutil.virtual_memory()
    logger.info(f"\nSystem Memory Status:")
    logger.info(f"Total: {memory.total / (1024**3):.2f} GB")
    logger.info(f"Available: {memory.available / (1024**3):.2f} GB")
    logger.info(f"Used: {memory.used / (1024**3):.2f} GB")
    logger.info(f"\nGPU Status:")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        )


def get_missing_files(save_directory):
    """
    Check which required files are missing from the directory
    Returns a tuple of (missing_tokenizer_files, missing_model_files)
    """
    tokenizer_files = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }

    model_files = {"config.json", "generation_config.json"}

    existing_files = (
        set(os.listdir(save_directory)) if os.path.exists(save_directory) else set()
    )

    # Check for model shards
    has_model_shards = any(
        f.startswith("model-00001-of-") and f.endswith(".safetensors")
        for f in existing_files
    )

    missing_tokenizer = tokenizer_files - existing_files
    missing_model = (model_files - existing_files) or (not has_model_shards)

    return missing_tokenizer, missing_model


def download_tokenizer(base_model, save_directory, logger, missing_files):
    """Download and save the tokenizer if not already present"""
    if not missing_files:
        logger.info("All tokenizer files present, skipping tokenizer download")
        return

    try:
        logger.info(
            f"Downloading tokenizer (missing files: {', '.join(missing_files)})..."
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            token=ACCESS_TOKEN,
        )

        logger.info("Saving tokenizer...")
        tokenizer.save_pretrained(save_directory, from_pt=True)
        logger.info("Tokenizer saved successfully")

    except Exception as e:
        logger.error(f"Error downloading tokenizer: {str(e)}")
        raise


def download_model(base_model, save_directory, logger, missing_files):
    """Download and save the model"""
    if not missing_files:
        logger.info("All model files present, skipping model download")
        return None

    try:
        # Get the missing files from environment variable
        missing_files_str = os.getenv("MISSING_FILES", "")
        logger.debug(f"Missing files string: {missing_files_str}")
        missing_files = missing_files_str.split() if missing_files_str else []
        logger.debug(f"Missing files: {missing_files}")

        logger.info(
            f"Downloading model (missing files: {', '.join(missing_files) if missing_files else 'all'})... This may take several minutes."
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # quantization_config=quant_config,
            token=ACCESS_TOKEN,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )

        logger.info(f"Model state before saving: {model.config}")
        logger.info(f"Available disk space: {shutil.disk_usage('/')}")
        logger.info(f"Memory usage before saving: {psutil.virtual_memory().percent}%")

        logger.info("Saving model... This may take several minutes.")
        model.save_pretrained(
            save_directory,
            # max_shard_size="1GB",
            from_pt=True,
            safe_serialization=True,
        )

        logger.info("Model saved successfully")
        return model

    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise


def download_and_save_model():
    """Main function to orchestrate the download process"""
    logger = setup_logging()
    BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    save_directory = "/models/mistral"
    os.makedirs(save_directory, exist_ok=True)

    check_system_resources(logger)
    logger.info("\nInitializing download process...")

    try:
        # Check what's missing
        missing_tokenizer, missing_model = get_missing_files(save_directory)

        if not (missing_tokenizer or missing_model):
            logger.info("All files present, no downloads needed")
            return

        # Download only what's needed
        if missing_tokenizer:
            download_tokenizer(BASE_MODEL, save_directory, logger, missing_tokenizer)

        if missing_model:
            download_model(BASE_MODEL, save_directory, logger, missing_model)

        logger.info(
            f"All missing files downloaded successfully to: {os.path.abspath(save_directory)}"
        )
        logger.info(
            "\nNote: Model has been saved in full precision format. It will be automatically quantized when loaded for inference"
        )

    except Exception as e:
        logger.error(f"\nError during model processing: {str(e)}")
        logger.error("\nFull error details:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires GPU support.")
        sys.exit(1)

    download_and_save_model()
