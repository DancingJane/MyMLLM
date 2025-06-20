import torch.nn as nn

from typing import Optional

from common.registry import registry
from model.projector import get_multimodal_projector
from model.llama.model import precompute_freqs_cis, Transformer
from model.nt.nt_model import NtModel
from model import BaseTokenizer, NtTokenizer, Llama3Tokenizer

import re
import torch
from typing import List, Union, Tuple, Optional

from common.utils import load_ckpt

def initialize_transformer(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def load_pretrained_weights(model, ckpt_path):
    # 加载原始权重
    pretrained_dict = torch.load(ckpt_path, map_location='cpu')
    
    # 移除不需要的前缀
    pretrained_dict = {k.replace('esm.', ''): v for k, v in pretrained_dict.items()}
    
    # 过滤掉lm_head相关参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('lm_head')}
    
    # 加载权重（strict=False允许部分加载）
    model.load_state_dict(pretrained_dict, strict=False)

# @registry.register_model(["llama1_with_nt", "llama2_with_nt"])
class LlamaWithNt(nn.Module):
    def __init__(self, config, model_ckpt_path:Optional[str]=None, multimodal_model_ckpt_path:Optional[str]=None):
        super().__init__()
        self.config = config
        self.nt_config = config.multimodal_model_config
        self.model = Transformer(config)
        self.multimodal_model = NtModel(self.nt_config)
        self.multimodal_projector = get_multimodal_projector(config)
        initialize_transformer(self.multimodal_projector)
        try:
            self.tokenizer = registry.get_tokenizer_class('llama3')(self.config.tokenizer)
            self.multimodal_tokenizer = NtTokenizer(self.config.multimodal_model_config.tokenizer)
        except Exception as e:
            print(e)
        self.global_rank = 0
        self.config.vocab_size = self.tokenizer.n_words

        if model_ckpt_path and multimodal_model_ckpt_path:
            if model_ckpt_path == multimodal_model_ckpt_path:  # 是同一个ckpt文件
                print("Loading full model checkpoint...")
                # full_state_dict = torch.load(model_ckpt_path, map_location='cpu')
                try:
                    # 方法1：使用 weights_only=False
                    full_state_dict = torch.load(model_ckpt_path, map_location='cpu', weights_only=False)
                    
                    # 方法2：或者使用安全全局变量
                    # from torch.serialization import add_safe_globals
                    # from deepspeed.runtime.fp16.loss_scaler import LossScaler
                    # add_safe_globals([LossScaler])
                    # full_state_dict = torch.load(model_ckpt_path, map_location='cpu')
                    
                    # 处理权重加载...
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
                    # 尝试方法3：提取纯模型权重
                    checkpoint = torch.load(model_ckpt_path, map_location='cpu', weights_only=False)
                    if 'module' in checkpoint:
                        full_state_dict = checkpoint['module']
                        torch.save(full_state_dict, 'temp_pure_weights.pt')
                        full_state_dict = torch.load('temp_pure_weights.pt', map_location='cpu')

                
                # 检查是否有model.前缀
                if any(k.startswith('model.') for k in full_state_dict.keys()):
                    # 处理Llama部分
                    llama_state_dict = {k.replace('model.', ''): v 
                                      for k, v in full_state_dict.items() 
                                      if k.startswith('model.')}
                    self.model.load_state_dict(llama_state_dict, strict=False)
                    
                    # 处理Nt部分
                    nt_state_dict = {k.replace('multimodal_model.', ''): v 
                                     for k, v in full_state_dict.items() 
                                     if k.startswith('multimodal_model.')}
                    self.multimodal_model.load_state_dict(nt_state_dict, strict=False)
                    
                    # 处理投影层
                    proj_state_dict = {k: v for k, v in full_state_dict.items() 
                                     if k.startswith('multimodal_projector.')}
                    self.multimodal_projector.load_state_dict(proj_state_dict, strict=False)
                else:
                    # 如果没有前缀，尝试直接加载
                    self.load_state_dict(full_state_dict, strict=False)
            else:
                # 原来的分开加载逻辑
                load_ckpt(model=self.model, ckpt_path=model_ckpt_path, rank=self.global_rank)
                # load_ckpt(model=self.multimodal_model, ckpt_path=multimodal_model_ckpt_path, rank=self.global_rank)
                load_pretrained_weights(self.multimodal_model, multimodal_model_ckpt_path)


        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads, 
            config.max_seq_len * 2,
            config.rope_theta
        )

    def forward(self):
        pass

    @torch.inference_mode()
    def generate(
        self,
        inputs: Union[List[str], str],
        device,
        output_len: int,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        assert hasattr(self, "tokenizer"), "Can not inference with out a provieded tokenizer"
        if isinstance(inputs, str):
            inputs = [inputs]
        bsz = len(inputs)
            
        input_embs = []    
        for input_sentence in inputs:
            t = []
            pos = 0
            pattern = r'[ACTG]{6,}'
            for match in re.finditer(pattern, input_sentence):
                start, end = match.span()
                if pos < start:
                    word_ids = torch.LongTensor(self.tokenizer.encode(input_sentence[pos:start])).to(device)
                    word_embs = self.model.tok_embeddings(word_ids)
                    t.append(word_embs)
                dna_ids = torch.LongTensor(self.multimodal_tokenizer.encode(input_sentence[start:end])).to(device)
                dna_embs = self.multimodal_model(dna_ids.unsqueeze(0))
                dna_embs = self.multimodal_projector(dna_embs)
                t.append(dna_embs.squeeze())
                pos = end
            if pos < len(input_sentence):
                word_ids = torch.LongTensor(self.tokenizer.encode(input_sentence[pos:start])).to(device)
                word_embs = self.model.tok_embeddings(word_ids)
                t.append(word_embs)
            input_embs.append(torch.cat(t, dim=0))

        params = self.config.multimodal_model_config

        min_prompt_len = min(t.shape[0] for t in input_embs)
        max_prompt_len = max(t.shape[0] for t in input_embs)
        total_len = output_len + max_prompt_len

        pad_id = self.model.tok_embeddings(torch.LongTensor([self.tokenizer.pad_id]).to(device)).squeeze()
        pad_input_embs = pad_id.unsqueeze(0).unsqueeze(0).expand(bsz, total_len, -1).clone()
        for i, input_emb in enumerate(input_embs):
            pad_input_embs[i, : len(input_emb)] = input_emb

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = (pad_input_embs != pad_id)[:, :, 0]

        # remove kv cache from attention attrs, for unified API
        caches_kv = None
        # caches_kv = []
        # for _ in range(params.num_hidden_layers):
        #     n_kv_heads = params.n_kv_heads if params.n_kv_heads else params.n_heads
        #     size = (bsz, params.max_seq_len, n_kv_heads,
        #             params.head_dim)
        #     dtype = params.get_dtype()
        #     cache_k = torch.zeros(size=size, dtype=dtype, device=device)
        #     cache_v = torch.zeros(size=size, dtype=dtype, device=device)
        #     caches_kv.append((cache_k, cache_v))

        out_tokens = [[] for i in range(bsz)]
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens=pad_input_embs, 
                                        start_pos=prev_pos, 
                                        freqs_cis=self.freqs_cis,
                                        atten_type='',
                                        caches_kv=caches_kv,
                                        is_embed=True)
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens=pad_input_embs[:, prev_pos:cur_pos], 
                                        start_pos=prev_pos, 
                                        freqs_cis=self.freqs_cis, 
                                        atten_type='',
                                        caches_kv=caches_kv)
            next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)

            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            for i, token in enumerate(next_token.tolist()):
                out_tokens[i].append(token)
            pad_input_embs[:, cur_pos] = self.model.tok_embeddings(next_token.to(torch.long))
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_words = []
        print(out_tokens)
        for i, sample in enumerate(out_tokens):
            # cut to max gen len
            if self.tokenizer.eos_id in sample:
                eos_idx = sample.index(self.tokenizer.eos_id)
                sample = sample[:eos_idx]
            # out_tokens.append(toks)
            out_words.append(self.tokenizer.decode(sample))
        return out_words

# @registry.register_model("llama3_with_nt")
class Llama3WithNt(LlamaWithNt):
    def __init__(self, config, tokenizer:Optional[str]=None):
        super().__init__(config)
        try:
            if tokenizer is None:
                self.tokenizer = Llama3Tokenizer(config.tokenizer)
            else:
                self.tokenizer = Llama3Tokenizer(tokenizer)
        except:
            pass


if __name__ == '__main__':
    # multimodal_model_ckpt_path = '/tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/pytorch_model.bin'
    # pretrained_dict = torch.load(multimodal_model_ckpt_path, map_location='cpu')
    # print("Pretrained contact_head shape:", pretrained_dict['contact_head.regression.weight'].shape)

    model_config = registry.get_model_config_class("llama3_with_nt_large")()
    model_config.tokenizer = '/tos-bjml-ai4agr/lijinzhe/BioModel/Meta-Llama-3-8B-Instruct/original/tokenizer.model'
    model_config.multimodal_model_config.tokenizer = '/tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/'
    # model_config.multimodal_model_config.vocab_size=4107
    # model_config.multimodal_model_config.num_hidden_layers=27
    # model_config.multimodal_model_config.hidden_size=1280
    # model_config.multimodal_model_config.intermediate_size=5120
    model_config.multimodal_model_config.vocab_size=4107
    model_config.multimodal_model_config.hidden_size=1024
    model_config.multimodal_model_config.intermediate_size=4096
    model_config.multimodal_model_config.num_hidden_layers=29
    model_config.multimodal_model_config.num_attention_heads=16
    model_config.multimodal_model_config.max_position_embeddings=2050

    # model_config.bert_config.device = 'cuda'
    model_ckpt_path = '/tos-bjml-ai4agr/lijinzhe/BioModel/Meta-Llama-3-8B-Instruct/original/consolidated.00.pth'
    multimodal_model_ckpt_path = '/tos-bjml-ai4agr/lijinzhe/BioModel/nucleotide-transformer/pytorch_model.bin'
    # model_ckpt_path = '/fs-computility/ai4agr/mazhe/MyMLLM/output_ckpt/multimodal_instruct_tuning_exp_20250611_nt_dna/final.ckpt'
    # multimodal_model_ckpt_path = '/fs-computility/ai4agr/mazhe/MyMLLM/output_ckpt/multimodal_instruct_tuning_exp_20250611_nt_dna/final.ckpt'

    test_model = LlamaWithNt(model_config, model_ckpt_path, multimodal_model_ckpt_path)
    test_model.to(torch.float16)
    test_model.to('cuda')
    result = test_model.generate('this is a test: ATCGATCGATCG', 
                   device=torch.device('cuda'),
                   output_len=10)
    print(result)