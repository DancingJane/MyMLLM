import torch.nn as nn
import torch
import re

from typing import Optional, List

from common.registry import registry
from model.projector import get_multimodal_projector
from model.llama.model import precompute_freqs_cis, Transformer
from model.dnabert.bert_model import BertModel
from model import BaseTokenizer, DnaBert2Tokenizer, Llama3Tokenizer

from liger_kernel.transformers import LigerCrossEntropyLoss, LigerFusedLinearCrossEntropyLoss

# 
def initialize_transformer(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

@registry.register_model(["llama1_with_bert", "llama2_with_bert"])
class LlamaWithBert(nn.Module):
    def __init__(self, config, tokenizer:Optional[str]=None, multimodal_tokenizer:Optional[str]=None):
        super().__init__()
        self.config = config
        bert_config = config.multimodal_model_config
        self.model = Transformer(config)
        self.multimodal_model = BertModel(bert_config)
        self.multimodal_projector = get_multimodal_projector(config)
        initialize_transformer(self.multimodal_projector)
        try:
            if tokenizer is None:
                self.tokenizer = BaseTokenizer(config.tokenizer)
            else:
                self.tokenizer = BaseTokenizer(tokenizer)
            if multimodal_tokenizer is None:
                self.multimodal_tokenizer = DnaBert2Tokenizer(registry.get_path("tokenizer_dnabert2"))
            else:
                self.multimodal_tokenizer = DnaBert2Tokenizer(multimodal_tokenizer)
        except:
            pass
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads, 
            config.max_seq_len * 2,
            config.rope_theta
        )

    def forward(self):
        pass

@registry.register_model("llama3_with_bert")
class Llama3WithBert(LlamaWithBert):
    def __init__(self, config, tokenizer:Optional[str]=None):
        super().__init__(config)
        try:
            if tokenizer is None:
                self.tokenizer = Llama3Tokenizer(config.tokenizer)
            else:
                self.tokenizer = Llama3Tokenizer(tokenizer)
        except:
            pass


# 增加🌟
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
# from transformers.models.bert.modeling_bert import BertModel

@registry.register_model("qwen2_with_bert")
class QwenWithBert(nn.Module):
    def __init__(self, config, tokenizer:Optional[str]=None, multimodal_tokenizer:Optional[str]=None):
        super().__init__()
        self.config = config
        bert_config = config.multimodal_model_config
        # import pdb
        # pdb.set_trace()
        self.model = Qwen2ForCausalLM(config)
        self.multimodal_model = BertModel(bert_config)
        self.multimodal_projector = get_multimodal_projector(config)
        # 训练的时候需要测试的时候不需要
        initialize_transformer(self.multimodal_projector)
        try:
            
            self.tokenizer = tokenizer
            self.multimodal_tokenizer = multimodal_tokenizer
        except:
            pass
        self.freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
            config.rope_theta
            # config.max_seq_len * 2,
            # config.rope_theta
        )


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
@registry.register_model("qwen3_with_bert")
class Qwen3WithBert(nn.Module):
    def __init__(self, config):
        super().__init__()

        # import torch.nn.init as init

        # # 临时替换 init 方法为 no-op, 调试的时候添加，用于加速构建
        # init.kaiming_uniform_ = lambda *args, **kwargs: None
        # init.uniform_ = lambda *args, **kwargs: None
        # init.normal_ = lambda *args, **kwargs: None

        self.text_config = config.text_config
        self.bio_config = config.multimodal_model_config
        # self.model = None
        self.model = Qwen3ForCausalLM(self.text_config)
        # self.model = AutoModelForCausalLM.from_config(self.text_config)
        # self.bio_model = BertModel(self.bio_config)
        self.bio_model = AutoModel.from_config(self.bio_config)
        self.multimodal_projector = nn.Linear(self.bio_config.hidden_size, self.text_config.hidden_size)
    
    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]                      # [B, L]
        dna_ids_list = kwargs["dna_ids_lists"]                # List[Tensor], 每个样本内多个 DNA 片段
        dna_start_pos_list = kwargs["dna_start_pos_lists"]    # List[List[int]], 每个样本内每段 DNA 的插入位置
        labels = kwargs["labels"]                            # [B, L]

        batch_size, seq_len = input_ids.shape
        hidden_states = self.model.get_input_embeddings()(input_ids)  # [B, L, D]
        hidden_dim = hidden_states.shape[-1]

        # 将 dna_ids_list[b][i] 全部打平
        flat_dna_ids = []
        mapping = []  # (batch_idx, start_pos, length)
        for b in range(batch_size):
            for i, start_pos in enumerate(dna_start_pos_list[b]):
                dna = dna_ids_list[b][i]
                flat_dna_ids.append(dna)
                mapping.append((b, start_pos, len(dna)))

        # 直接堆叠成 tensor，因为所有 dna 序列长度一致
        padded_dna = torch.stack(flat_dna_ids, dim=0).to(hidden_states.device)  # [N, fixed_len]
        dna_embeddings = self.bio_model(padded_dna)[0]  # 直接返回 [N, L_dna, hidden_size]
        proj_out = self.multimodal_projector(dna_embeddings)
        # scatter 回去
        for i, (b, start_pos, length) in enumerate(mapping):
            hidden_states[b, start_pos:start_pos+length, :] = proj_out[i, :length, :]

        # 进入 Qwen
        outputs = self.model(
            inputs_embeds=hidden_states,
            labels=labels
        )
        # outputs.logits
        return outputs.loss, None

    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,               # [B, L]
        dna_ids_lists: List[List[torch.LongTensor]],   # List (batch) of List (segments) of [seg_len]
        dna_start_pos_lists: List[List[int]],         # List (batch) of start positions
        attention_mask: Optional[torch.LongTensor] = None,  # [B, L], 若有
        **generate_kwargs                           # e.g. max_length=..., num_beams=..., do_sample=...
    ) -> torch.LongTensor:
        """
        融合 DNA-BERT 提取的多模态 embedding，使用 Qwen3 的 generate 接口生成序列。

        返回：
            output_ids: [B, T_gen] 生成的 token id 序列
        """
        device = input_ids.device

        # 1. 计算初始 token embedding
        hidden_states = self.model.get_input_embeddings()(input_ids)  # [B, L, D]

        # 2. 将所有 DNA 片段打平并到同设备
        flat_dna = []
        mapping = []  # (batch_idx, start_pos, length)
        for b, (dna_list, pos_list) in enumerate(zip(dna_ids_lists, dna_start_pos_lists)):
            for dna_ids, start in zip(dna_list, pos_list):
                flat_dna.append(dna_ids.to(device))
                mapping.append((b, start, dna_ids.size(0)))

        padded_dna = torch.stack(flat_dna, dim=0)  # [N, seg_len]

        # 3. 用 DNA-BERT 提取 embedding 并投射
        dna_emb = self.bio_model(padded_dna)[0]                     # [N, seg_len, H_bio]
        proj_emb = self.multimodal_projector(dna_emb)               # [N, seg_len, H_text]

        # 4. 把投射后的 embedding 插回到 hidden_states
        for i, (b, start, length) in enumerate(mapping):
            hidden_states[b, start:start+length, :] = proj_emb[i, :length, :]

        # 5. 准备 attention_mask（如未传则全1）
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        # 6. 调用底层 generate（传入 inputs_embeds 而非 input_ids）
        ml = 2048
        temperature = 0.8
        top_p=0.95
        output_ids = self.model.generate(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            max_length=ml,
            temperature=temperature,
            top_p=top_p
        )
        return output_ids
    


if __name__ == "__main__":
    model_config = registry.get_model_config_class("qwen3_with_bert")
