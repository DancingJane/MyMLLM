import torch
from typing import Optional
from dataclasses import dataclass

from common.registry import registry
from common.utils import STR_DTYPE_TO_TORCH_DTYPE

from transformers.models.bert.configuration_bert import BertConfig

@registry.register_model_config("dnabert2")
def get_dnabert2():
    config = BertConfig(vocab_size=4096, 
                      n_projector_layers=3)
    config.atten_type = 'flash_atten'
    config.projector_type = 'linear'
    config.mode = 'pool'
    config.l_output  = None
    config.use_lengths = False
    config.lengths = None
    config.tokenizer = ''
    config.add_pooling_layer = False
    config.alibi_starting_size = 512
    config.pad_token_id = 0
    return config

@registry.register_model_config("llama1_with_bert_large")
def get_llama_hyena_config_large():
    model_config = registry.get_model_config_class("llama1_7b")()
    model_config.multimodal_model_config = get_dnabert2()
    return model_config

@registry.register_model_config("llama2_with_bert_large")
def get_llama_hyena_config_large():
    model_config = registry.get_model_config_class("llama2_7b")()
    model_config.multimodal_model_config = get_dnabert2()
    return model_config

@registry.register_model_config("llama3_with_bert_large")
def get_llama_hyena_config_large():
    model_config = registry.get_model_config_class("llama3_8b")()
    model_config.multimodal_model_config = get_dnabert2()
    return model_config

# 🌟增加修改
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, Qwen3ForCausalLM, AutoConfig, BertConfig
@registry.register_model_config("qwen2_with_bert_large")
def get_llama_hyena_config_large():
    model_config = registry.get_model_config_class("qwen2_config")()
    model_config.multimodal_model_config = get_dnabert2()
    return model_config

@registry.register_model_config("qwen3_with_bert_large")
def get_qwen3_bert_config_large(text_model_path, bio_model_path):
    model_config = AutoConfig.from_pretrained(text_model_path, trust_remote_code=True)
    model_config.text_config = AutoConfig.from_pretrained(text_model_path, trust_remote_code=True)
    model_config.multimodal_model_config = BertConfig.from_pretrained(bio_model_path, trust_remote_code=True)
    return model_config

