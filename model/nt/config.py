import torch
from typing import Optional
from dataclasses import dataclass

from common.registry import registry
from common.utils import STR_DTYPE_TO_TORCH_DTYPE

@dataclass
class NtConfig: #TODO: to set alll the parameters below
    vocab_size: int = 30522 
    n_projector_layers: int = 1
    dim: int = 768 
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    use_cache: bool = True
    classifier_dropout: Optional[float] = None
    alibi_starting_size: int = 512
    attention_probs_dropout_prob: float = 0.0
    dtype: str = 'float32'
    atten_type: str = 'flash_atten'
    projector_type: str = 'linear'
    mode: str = 'pool'
    l_output: Optional[int] = None
    use_lengths: bool = False
    lengths: Optional[int] = None
    tokenizer: Optional[str] = ''
    add_pooling_layer: bool = False
    def get_dtype(self) -> Optional[torch.dtype]:
        """Gets the torch dtype from the config dtype string."""
        return STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)

@registry.register_model_config("nt")
def get_nt(): #TODO: set the parameters below
    return NtConfig(vocab_size=4096, 
                      n_projector_layers=3)


@registry.register_model_config("llama3_with_nt_large")
def get_llama_nt_config_large():
    model_config = registry.get_model_config_class("llama3_8b")()
    model_config.multimodal_model_config = get_nt()
    return model_config

