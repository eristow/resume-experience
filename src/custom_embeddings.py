"""
custom_embeddings.py

This module provides a custom embeddings class for generating embeddings using a pre-trained language model.
"""

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from langchain.embeddings.base import Embeddings
from logger import setup_logging
import uuid
import threading


class ModelManager:
    """
    Singleton class to manage model resources across sessions
    """

    logger = setup_logging()
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance.model = None
                    cls._instance.tokenizer = None
                    cls._instance.use_count = 0
                    cls._instance.model_lock = threading.RLock()
        return cls._instance


class CustomEmbeddings(Embeddings):
    """
    CustomEmbeddings class for generating embeddings using a pre-trained language model.
    """

    logger = setup_logging()

    def __init__(self, model_name):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self._model = None
        # self._tokenizer = None
        self.manager = ModelManager()
        self.logger.info(f"Created new CustomEmbeddings instance")

    def get_model(self):
        """Get or create model instance"""
        # if self._model is None:
        #     self.logger.info(f"Loading model...")
        #     quant_config = BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_compute_dtype=torch.float16,
        #         bnb_4bit_use_double_quant=True,
        #     )

        #     # Create a device map that allows CPU offloading
        #     device_map = "auto"
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()

        #         total_memory = torch.cuda.get_device_properties(0).total_memory
        #         memory_allocated = torch.cuda.memory_allocated(0)
        #         available_memory = total_memory - memory_allocated

        #         self.logger.info(f"GPU Total Memory: {total_memory / 1024**2:.2f}MB")
        #         self.logger.info(f"GPU Memory Allocated: {memory_allocated / 1024**2:.2f}MB")
        #         self.logger.info(f"GPU Available Memory: {available_memory / 1024**2:.2f}MB")

        #         if available_memory < 6 * 1024**3:
        #             self.logger.info("Limited GPU memory available, using CPU offload")
        #             device_map = {
        #                 "transformer.word_embeddings": 0,
        #                 "transformer.h": "cpu",
        #                 "transformer.ln_f": 0,
        #                 "lm_head": 0,
        #             }
        #         else:
        #             device_map = "auto"
        #             self.logger.info(
        #                 "Sufficient GPU memory available, using auto device map"
        #             )

        #     try:
        #         self._model = AutoModel.from_pretrained(
        #             self.model_name,
        #             quantization_config=quant_config,
        #             device_map=device_map,
        #             trust_remote_code=True,
        #             torch_dtype=torch.float16,
        #         )
        #         self._model.eval()

        #         # Verify model device placement
        #         # for name, param in self._model.named_parameters():
        #         #     self.logger.info(f"Parameter {name} is on device: {param.device}")

        #     except Exception as e:
        #         self.logger.error(f"Error loading model: {e}")
        #         self.logger.info("Falling back to CPU-only mode")
        #         self._model = AutoModel.from_pretrained(
        #             self.model_name,
        #             device_map="cpu",
        #             trust_remote_code=True,
        #             torch_dtype=torch.float32,
        #         )
        #         self._model.eval()

        # return self._model

        with self.manager.model_lock:
            if self.manager.model is None:
                self.logger.info(f"Loading model...")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                        total_memory = torch.cuda.get_device_properties(0).total_memory
                        memory_allocated = torch.cuda.memory_allocated(0)
                        available_memory = total_memory - memory_allocated

                        self.logger.info(
                            f"GPU Total Memory: {total_memory / 1024**2:.2f}MB"
                        )
                        self.logger.info(
                            f"GPU Memory Allocated: {memory_allocated / 1024**2:.2f}MB"
                        )
                        self.logger.info(
                            f"GPU Available Memory: {available_memory / 1024**2:.2f}MB"
                        )

                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )

                    # Create device map based on available GPU memory
                    if (
                        torch.cuda.is_available() and available_memory > 6 * 1024**3
                    ):  # If more than 6GB available
                        self.logger.info("Using full GPU acceleration")
                        device_map = "auto"
                    else:
                        self.logger.info(
                            "Using partial GPU acceleration with CPU offload"
                        )
                        device_map = {
                            "transformer.word_embeddings": 0,
                            "transformer.h": ["0"] * 8
                            + ["cpu"] * 24,  # First 8 layers on GPU
                            "transformer.ln_f": 0,
                            "lm_head": 0,
                        }

                    self.manager.model = AutoModel.from_pretrained(
                        self.model_name,
                        quantization_config=quant_config,
                        device_map=device_map,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                    )
                    self.manager.model.eval()

                except Exception as e:
                    self.logger.error(f"Error loading model with GPU: {e}")
                    self.logger.info("Falling back to CPU-only mode")
                    self.manager.model = AutoModel.from_pretrained(
                        self.model_name,
                        device_map="cpu",
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                    )
                    self.manager.model.eval()

            self.manager.use_count += 1
            return self.manager.model

    def get_tokenizer(self):
        # if self._tokenizer is None:
        #     self.logger.info(f"Loading tokenizer...")
        #     self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #     self._tokenizer.pad_token = self._tokenizer.eos_token
        # return self._tokenizer

        with self.manager.model_lock:
            if self.manager.tokenizer is None:
                self.logger.info(f"Loading tokenizer...")
                self.manager.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.manager.tokenizer.pad_token = self.manager.tokenizer.eos_token
            return self.manager.tokenizer

    def embed_func(self, inputs):
        """
        Embeds the given inputs using the model's tokenizer and returns the embedding.
        """
        # model = self.get_model()
        # tokenizer = self.get_tokenizer()

        # self.logger.info(f"Embeddings processing input: {inputs[:50]}...")

        # tokens = tokenizer(
        #     inputs,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True,
        #     max_length=512,
        # )

        # # Get model's device and dtype
        # model_param = next(model.parameters())
        # device = model_param.device
        # dtype = model_param.dtype
        # self.logger.info(f"Model is on device {device} with dtype {dtype}")

        # # Convert tokens to the correct dtype
        # processed_tokens = {
        #     "input_ids": tokens["input_ids"].to(device=device, dtype=torch.long),
        #     "attention_mask": tokens["attention_mask"].to(device=device, dtype=dtype),
        # }

        # try:
        #     with torch.no_grad():
        #         with torch.amp.autocast(device_type="cuda", dtype=dtype):
        #             outputs = model(**processed_tokens)
        #         embeddings = outputs.last_hidden_state.mean(dim=1)

        #     return embeddings.cpu().numpy().flatten().tolist()

        # except RuntimeError as e:
        #     self.logger.error(f"Runtime error during embedding: {e}")
        #     # Fallback to CPU with float32
        #     model.cpu()
        #     processed_tokens = {
        #         "input_ids": tokens["input_ids"].cpu().long(),
        #         "attention_mask": tokens["attention_mask"].cpu().float(),
        #     }

        #     with torch.no_grad():
        #         outputs = model(**processed_tokens)
        #         embeddings = outputs.last_hidden_state.mean(dim=1)

        #     return embeddings.numpy().flatten().tolist()

        with self.manager.model_lock:
            model = self.get_model()
            tokenizer = self.get_tokenizer()

            self.logger.info(f"Embeddings processing input: {inputs[:50]}...")

            tokens = tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Get model's device and dtype
            model_param = next(model.parameters())
            device = model_param.device
            dtype = model_param.dtype
            self.logger.info(f"Model is on device {device} with dtype {dtype}")

            # Process on the same device as the model
            processed_tokens = {
                "input_ids": tokens["input_ids"].to(device=device, dtype=torch.long),
                "attention_mask": tokens["attention_mask"].to(
                    device=device, dtype=dtype
                ),
            }

            try:
                with torch.no_grad():
                    if device.type == "cuda":
                        with torch.amp.autocast(device_type="cuda", dtype=dtype):
                            outputs = model(**processed_tokens)
                    else:
                        outputs = model(**processed_tokens)

                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    return embeddings.cpu().numpy().flatten().tolist()

            except RuntimeError as e:
                self.logger.error(f"Runtime error during embedding: {e}")
                # Fallback to CPU if GPU runs out of memory
                model.cpu()
                processed_tokens = {
                    "input_ids": tokens["input_ids"].cpu(),
                    "attention_mask": tokens["attention_mask"].cpu(),
                }

                with torch.no_grad():
                    outputs = model(**processed_tokens)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.numpy().flatten().tolist()

    def embed_documents(self, texts):
        """Generate embeddings for a batch of documents"""
        batch_size = 8
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            self.logger.info(f"Embeddings processing batch {i//batch_size + 1}")
            batch_embeddings = [self.embed_func(text) for text in batch]
            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_query(self, text):
        """Generate embeddings for a query"""
        return self.embed_func(text)

    def cleanup(self):
        """Cleanup model resources"""
        # if self._model is not None:
        #     try:
        #         self._model.cpu()

        #         del self._model
        #         self._model = None

        #         if torch.cuda.is_available():
        #             torch.cuda.empty_cache()
        #     except Exception as e:
        #         self.logger.error(f"Error during model cleanup: {e}")
        # if self._tokenizer is not None:
        #     try:
        #         del self._tokenizer
        #         self._tokenizer = None
        #     except Exception as e:
        #         self.logger.error(f"Error during tokenizer cleanup: {e}")
        # self.logger.info("Cleaned up model and tokenizer resources")

        with self.manager.model_lock:
            self.manager.use_count -= 1
            self.logger.info(f"Decreased use count to {self.manager.use_count}")
            if self.manager.use_count <= 0:
                if self.manager.model is not None:
                    try:
                        self.manager.model.cpu()
                        del self.manager.model
                        self.manager.model = None

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            self.logger.info("Cleared CUDA cache")

                    except Exception as e:
                        self.logger.error(f"Error during model cleanup: {e}")

                if self.manager.tokenizer is not None:
                    try:
                        del self.manager.tokenizer
                        self.manager.tokenizer = None
                    except Exception as e:
                        self.logger.error(f"Error during tokenizer cleanup: {e}")

                self.manager.use_count = 0
                self.logger.info("Cleaned up model and tokenizer resources")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
