import torch
from typing import Optional
from dataclasses import dataclass
from common.registry import registry
from common.utils import STR_DTYPE_TO_TORCH_DTYPE

@dataclass
class NtConfig:
    # 核心参数（与 BertConfig 对齐）
    vocab_size: int = 32000  # ESM 默认词汇表大小
    dim: int = 768           # hidden_size
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1026  # ESM 特有
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 1               # ESM 的 <pad> token
    mask_token_id: int = 32             # ESM 的 <mask> token
    position_embedding_type: str = "absolute"
    use_cache: bool = True
    
    # MyTransformer 项目特有参数
    dtype: str = 'float32'
    atten_type: str = 'flash_atten'
    projector_type: str = 'linear'
    mode: str = 'pool'
    n_projector_layers: int = 1
    l_output: Optional[int] = None
    use_lengths: bool = False
    tokenizer: Optional[str] = 'esm'    # 指定 tokenizer 类型

    def get_dtype(self) -> Optional[torch.dtype]:
        return STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)

# 注册 ESM 配置
@registry.register_model_config("nt")
def get_nt_config():
    return NtConfig()

# 多模态组合配置（示例：ESM + LLaMA）
@registry.register_model_config("llama_with_nt")
def get_llama_nt_config():
    model_config = registry.get_model_config_class("llama3_8b")()  # 假设项目已注册 LLaMA
    model_config.multimodal_model_config = get_nt_config()
    return model_config

@registry.register_model_config("llama3_with_nt_large")
def get_llama_nt_config_large():
    model_config = registry.get_model_config_class("llama3_8b")()
    model_config.multimodal_model_config = get_nt_config()
    return model_config