"""
custom_embeddings.py

This module provides a custom embeddings class for generating embeddings using a pre-trained language model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.embeddings.base import Embeddings


class CustomEmbeddings(Embeddings):
    """
    CustomEmbeddings class for generating embeddings using a pre-trained language model.

    Args:
        model_name (str): The name or path of the pre-trained language model.

    Attributes:
        model (AutoModelForCausalLM): The pre-trained language model for generating embeddings.
        tokenizer (AutoTokenizer): The tokenizer for tokenizing input text.
    """

    def __init__(self, model_name):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=True, quantization_config=quant_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def embed_func(self, inputs):
        """
        Embeds the given inputs using the model's tokenizer and returns the embedding.

        Args:
            inputs (str): The input text to be embedded.

        Returns:
            list: The embedding of the input text.
        """

        tokens = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            outputs = self.model(**tokens)

        if hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states
        elif hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            raise AttributeError(
                "The model's output does not have 'hidden_states' or 'last_hidden_state'."
            )

        embedding = hidden_states[-1].mean(dim=1).cpu().numpy().flatten().tolist()
        return embedding

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            embedding = self.embed_func(text)
            embeddings.append(embedding)

        return embeddings

    def embed_query(self, text):
        return self.embed_func(text)
