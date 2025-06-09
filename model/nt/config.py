import torch

from common.registry import registry
from common.utils import STR_DTYPE_TO_TORCH_DTYPE


from dataclasses import asdict, dataclass
from typing import Optional

from transformers import PretrainedConfig, logging

logger = logging.get_logger(__name__)


@dataclass
class NtConfig(PretrainedConfig):
    model_type = "esm"

    def __init__(
        self,
        vocab_size=None,
        mask_token_id=None,
        pad_token_id=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1026,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        use_cache=True,
        emb_layer_norm_before=None,
        token_dropout=False,
        is_folding_model=False,
        esmfold_config=None,
        vocab_list=None,
        add_bias_fnn=True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id, mask_token_id=mask_token_id, **kwargs
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.emb_layer_norm_before = emb_layer_norm_before
        self.token_dropout = token_dropout
        self.is_folding_model = is_folding_model
        # Arguments needed for Dalmatian
        self.add_bias_fnn = add_bias_fnn

        # Arguments added to be compatible  with original MyTransformer
        self.dtype = 'float32'
        self.dim = self.hidden_size
        self.n_projector_layers = 1
        self.hidden_act = "gelu"
        self.type_vocab_size = 2
        self.alibi_starting_size = 512
        self.atten_type = 'flash_atten'
        self.projector_type = 'linear'
        self.mode = 'pool'
        self.l_output = None
        self.use_lengths = False
        self.lengths = None
        self.tokenizer = ''
        self.add_pooling_layer = False

        if is_folding_model:
            if esmfold_config is None:
                logger.info(
                    "No esmfold_config supplied for folding model, using default values."
                )
                esmfold_config = EsmFoldConfig()
            elif isinstance(esmfold_config, dict):
                esmfold_config = EsmFoldConfig(**esmfold_config)
            self.esmfold_config = esmfold_config
            if vocab_list is None:
                logger.warning(
                    "No vocab_list supplied for folding model, assuming the ESM-2 vocabulary!"
                )
                self.vocab_list = get_default_vocab_list()
            else:
                self.vocab_list = vocab_list
        else:
            self.esmfold_config = None
            self.vocab_list = None
        if self.esmfold_config is not None and getattr(
            self.esmfold_config, "use_esm_attn_map", False
        ):
            raise ValueError(
                "The HuggingFace port of ESMFold does not support use_esm_attn_map at this time!"
            )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()
        if isinstance(self.esmfold_config, EsmFoldConfig):
            output["esmfold_config"] = self.esmfold_config.to_dict()
        return output

    def get_dtype(self) -> Optional[torch.dtype]:
        """Gets the torch dtype from the config dtype string."""
        return STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)

@registry.register_model_config("nt")
def get_nt_config():
    return NtConfig(vocab_size=4107, mask_token_id=2, pad_token_id=1, emb_layer_norm_before=False)

@registry.register_model_config("llama1_with_nt_large")
def get_llama_hyena_config_large():
    model_config = registry.get_model_config_class("llama1_7b")()
    model_config.multimodal_model_config = get_nt_config()
    return model_config

@registry.register_model_config("llama2_with_nt_large")
def get_llama_hyena_config_large():
    model_config = registry.get_model_config_class("llama2_7b")()
    model_config.multimodal_model_config = get_nt_config()
    return model_config

@registry.register_model_config("llama3_with_nt_large")
def get_llama_hyena_config_large():
    model_config = registry.get_model_config_class("llama3_8b")()
    model_config.multimodal_model_config = get_nt_config()
    return model_config