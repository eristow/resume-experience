# adapt the process to use this code to correctly download the model locally: #switch out hugging face address with location of model on local machine
import os

# Load Hugging Face token from environment
from token import ACCESS_TOKEN  # Import the access token from tools.py


# Switch out the model address to the location on your local machine
# This script will download the model to the specified location
# This is a temporary fix to get the model to work
# Download on local machine

import torch

torch.cuda.empty_cache()

from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

device = "cuda"  # the device to load the model onto

# Quantization configuration
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_qant=True,
)

# Model and tokenizer loading
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=quant_config
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Saving the model locally
save_directory = r"./mistral"
os.makedirs(save_directory, exist_ok=True)
model.save_pretrained(save_directory, from_pt=True)
tokenizer.save_pretrained(save_directory, from_pt=True)

print("Model and tokenizer have been saved locally.", save_directory)
