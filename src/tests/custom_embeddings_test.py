import pytest
from unittest.mock import Mock, patch
import torch
import numpy as np
from custom_embeddings import CustomEmbeddings


# TODO: figure out how to not need a real model to test this
# class TestCustomEmbeddings:
#     model_name = "mistralai/Mistral-7B-Instruct-v0.3"

#     @pytest.fixture
#     def mock_model(self):
#         model = Mock()
#         # Mock hidden states output
#         hidden_states = torch.rand(1, 5, 768)  # batch_size=1, seq_len=5, hidden_dim=768
#         model.return_value = Mock(hidden_states=(hidden_states,))
#         return model

#     @pytest.fixture
#     def mock_tokenizer(self):
#         tokenizer = Mock()
#         tokenizer.return_value = {
#             "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
#             "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
#         }
#         tokenizer.pad_token = None
#         tokenizer.eos_token = "[EOS]"
#         return tokenizer

#     @pytest.fixture
#     def embeddings(self, mock_model, mock_tokenizer):
#         with patch("custom_embeddings.AutoModelForCausalLM") as model_cls, patch(
#             "custom_embeddings.AutoTokenizer"
#         ) as tokenizer_cls:

#             model_cls.from_pretrained.return_value = mock_model
#             tokenizer_cls.from_pretrained.return_value = mock_tokenizer

#             return CustomEmbeddings(self.model_name)

#     def test_init(self):
#         embeddings = CustomEmbeddings(self.model_name)
#         assert embeddings.model_name == self.model_name
#         assert embeddings._model is None
#         assert embeddings._tokenizer is None

#     def test_get_model(self, embeddings, mock_model):
#         model = embeddings.get_model(self.model_name, use_token=True)
#         assert model == mock_model
#         # Test singleton pattern
#         model2 = embeddings.get_model(self.model_name, use_token=True)
#         assert model2 is model

#     def test_get_tokenizer(self, embeddings, mock_tokenizer):
#         tokenizer = embeddings.get_tokenizer(self.model_name)
#         assert tokenizer == mock_tokenizer
#         assert tokenizer.pad_token == tokenizer.eos_token
#         # Test singleton pattern
#         tokenizer2 = embeddings.get_tokenizer(self.model_name)
#         assert tokenizer2 is tokenizer

#     def test_embed_func(self, embeddings):
#         test_input = "test text"
#         embedding = embeddings.embed_func(test_input)

#         assert isinstance(embedding, list)
#         # Check if embedding is a non-empty list of floats
#         assert len(embedding) > 0
#         assert all(isinstance(x, float) for x in embedding)

#     def test_embed_documents(self, embeddings):
#         texts = ["text1", "text2", "text3"]
#         embeddings_list = embeddings.embed_documents(texts)

#         assert isinstance(embeddings_list, list)
#         assert len(embeddings_list) == len(texts)
#         assert all(isinstance(emb, list) for emb in embeddings_list)
#         assert all(isinstance(x, float) for emb in embeddings_list for x in emb)

#     def test_embed_query(self, embeddings):
#         query = "test query"
#         embedding = embeddings.embed_query(query)

#         assert isinstance(embedding, list)
#         assert len(embedding) > 0
#         assert all(isinstance(x, float) for x in embedding)

#     @pytest.mark.parametrize("output_type", ["hidden_states", "last_hidden_state"])
#     def test_different_model_outputs(self, output_type, mock_tokenizer):
#         # Test both hidden_states and last_hidden_state outputs
#         mock_output = Mock()
#         hidden_state = torch.rand(1, 5, 768)

#         if output_type == "hidden_states":
#             setattr(mock_output, "hidden_states", (hidden_state,))
#         else:
#             setattr(mock_output, "last_hidden_state", hidden_state)

#         mock_model = Mock(return_value=mock_output)

#         with patch("custom_embeddings.AutoModelForCausalLM") as model_cls, patch(
#             "custom_embeddings.AutoTokenizer"
#         ) as tokenizer_cls:

#             model_cls.from_pretrained.return_value = mock_model
#             tokenizer_cls.from_pretrained.return_value = mock_tokenizer

#             embeddings = CustomEmbeddings(self.model_name)
#             embedding = embeddings.embed_func("test")

#             assert isinstance(embedding, list)
#             assert len(embedding) > 0

#     def test_missing_output_attributes(self, mock_tokenizer):
#         # Test error when model output has neither hidden_states nor last_hidden_state
#         mock_output = Mock()
#         mock_model = Mock(return_value=mock_output)

#         with patch("custom_embeddings.AutoModelForCausalLM") as model_cls, patch(
#             "custom_embeddings.AutoTokenizer"
#         ) as tokenizer_cls:

#             model_cls.from_pretrained.return_value = mock_model
#             tokenizer_cls.from_pretrained.return_value = mock_tokenizer

#             embeddings = CustomEmbeddings(self.model_name)

#             with pytest.raises(AttributeError):
#                 embeddings.embed_func("test")
