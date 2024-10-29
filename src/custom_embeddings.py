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

    _model = None
    _tokenizer = None

    def __init__(self, model_name):
        self.model_name = model_name

    @classmethod
    def get_model(cls, model_name):
        """Singleton pattern to reuse model across instances"""
        if cls._model is None:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            cls._model = AutoModelForCausalLM.from_pretrained(
                model_name, output_hidden_states=True, quantization_config=quant_config
            )
        return cls._model

    @classmethod
    def get_tokenizer(cls, model_name):
        """Singleton pattern to reuse tokenizer across instances"""
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls._tokenizer.pad_token = cls._tokenizer.eos_token
        return cls._tokenizer

    def embed_func(self, inputs):
        """
        Embeds the given inputs using the model's tokenizer and returns the embedding.

        Args:
            inputs (str): The input text to be embedded.

        Returns:
            list: The embedding of the input text.
        """

        model = self.get_model(self.model_name)
        tokenizer = self.get_tokenizer(self.model_name)

        tokens = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**tokens)

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
