import os
import json
import argparse
from argparse import Namespace
from functools import partial
import re

import torch
import pandas as pd
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from model import *
from common.utils import set_random_seed, load_ckpt, set_default_tensor_type
from common.registry import registry



def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal LLM Inference (DNA + Text)")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed inference, ignored if single GPU")
    parser.add_argument("--pretrained_ckpt", type=str, help="Path to pretrained language model checkpoint")
    parser.add_argument("--dna_ckpt", type=str, help="Path to DNA modality checkpoint (if any)")
    parser.add_argument("--trained_ckpt", type=str, help="Path to trained model parameters (base model)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Key name for model class in registry (e.g., qwen2_with_bert)")
    parser.add_argument("--model_variant", type=str, required=True,
                        help="Variant of the model (e.g., large, 7b, 8b)")
    parser.add_argument("--text_tokenizer", type=str, required=True,
                        help="Path or name for text tokenizer (e.g., HF Qwen)")
    parser.add_argument("--dna_tokenizer", type=str, required=True,
                        help="Path for DNA tokenizer (e.g., DNABERT-2)")
    parser.add_argument("--prompt", type=str, help="Single-prompt for interactive inference")
    parser.add_argument("--dataset_path", type=str,
                        help="Path to dataset file (jsonl/csv/xlsx/json) for batch inference. Must include column 'input'.")
    parser.add_argument("--output_path", type=str, default="./results",
                        help="Directory to save batch inference outputs")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum tokens to generate per example")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for dataset inference")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--quant", action="store_true",
                        help="Apply quantization settings from config")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu","cuda"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class MultiModalInfer:
    def __init__(self, args):
        self.args = args
        set_random_seed(args.seed)
        self.device = torch.device("cuda:0")

        # Load tokenizers
        self.text_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
        self.dna_tokenizer = AutoTokenizer.from_pretrained(args.dna_ckpt)

        # Load and initialize model
        config_type = '_'.join([args.model_name, args.model_variant])
        self.model_config = registry.get_model_config_class(config_type)(
            args.pretrained_ckpt, args.dna_ckpt
        )
        self._load_model()

    def _load_model(self):
        args = self.args
        # Apply basic config overrides
        self.model_config.quant = args.quant
        self.model_config.device = args.device
        self.model_config.max_len = 1024
        self.model_config.multimodal_encode_fp32 = False
        self.model_config.multimodal_k_tokens = 64

        with set_default_tensor_type(self.model_config.text_config.torch_dtype):
            model_cls = registry.get_model_class(self.args.model_name)
            self.model = model_cls(self.model_config)

            # load text LM weights
            text_lm = AutoModelForCausalLM.from_pretrained(
                self.args.pretrained_ckpt, trust_remote_code=True
            )
            self.model.model.load_state_dict(text_lm.state_dict())

            # load DNA-BERT weights (strict=False 允许部分加载)
            dna_lm = AutoModel.from_pretrained(
                self.args.dna_ckpt,
                config=self.model_config.multimodal_model_config,
                trust_remote_code=True
            )
            self.model.bio_model.load_state_dict(dna_lm.state_dict(), strict=False)

            # ===== 在加载完 text_lm 和 dna_lm 之后 =====
            if args.trained_ckpt:
                # 1) 把文件读进来
                ckpt = torch.load(
                    args.trained_ckpt,
                    map_location="cpu",
                    weights_only=False   # <— 加上这一句
                )
                state_dict = ckpt.get("model_state_dict", ckpt)
                # 3) 加载到你的自定义模型上
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded trained ckpt from {args.trained_ckpt}")
                if missing:
                    print(" ⚠️ Missing keys when loading trained ckpt:", missing)
                if unexpected:
                    print(" ⚠️ Unexpected keys when loading trained ckpt:", unexpected)

        self.model.to(self.device).eval()

    def process_sample(self, sample_text: str):
        """
        将一条包含 <dna>…<dna> 标签的文本，拆分为
        - input_ids: token 化后（含 pad 占位 DNA 段）
        - attention_mask
        - dna_ids_lists: 多段 DNA 的 token 序列 list
        - dna_start_pos_lists: 对应段的插入位置 list
        """
        text = sample_text
        # 抽出所有的 DNA 段
        pattern = r'<dna>(.*?)<dna>'
        matches = re.findall(pattern, text)
        # 去掉标签保留纯文本
        clean_text = re.sub(pattern, '', text)

        # 首先 tokenize 文本；后面再 overwrite DNA 段
        txt = self.text_tokenizer(clean_text, add_special_tokens=False)
        input_ids = txt["input_ids"].copy()
        attention_mask = [1] * len(input_ids)

        # 记录每段 DNA 处理后插入的位置
        dna_ids_lists, dna_start_pos_lists = [], []
        cursor = 0
        # 逐段查找原始位置，实际插入位置按 clean_text 的 token 长度推进
        for dna_seq in matches:
            # 找到这段 dna 在 clean_text 中应插入的位置（按字符）
            # 为简化示例，按顺序插在当前 cursor 后
            pos_tokens = len(input_ids)
            dna_tok = self.dna_tokenizer(
                dna_seq, add_special_tokens=False
            )["input_ids"]
            # 统一长度到 model_config.multimodal_k_tokens
            k = self.model_config.multimodal_k_tokens
            if len(dna_tok) < k:
                dna_tok = dna_tok + [self.dna_tokenizer.pad_token_id] * (k - len(dna_tok))
            else:
                dna_tok = dna_tok[:k]

            # 记录并用 text pad_token 占位
            dna_ids_lists.append(dna_tok)
            dna_start_pos_lists.append(pos_tokens)
            input_ids = input_ids[:pos_tokens] + \
                        [self.text_tokenizer.pad_token_id]*k + \
                        input_ids[pos_tokens:]
            attention_mask = attention_mask[:pos_tokens] + [1] * k + attention_mask[pos_tokens:]

        # 加上 eos
        input_ids.append(self.text_tokenizer.eos_token_id)
        attention_mask.append(1)

        # 限长 & pad 到 max_len
        max_len = self.model_config.max_len
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
        else:
            pad_len = max_len - len(input_ids)
            input_ids += [self.text_tokenizer.pad_token_id]*pad_len
            attention_mask += [0]*pad_len

        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "dna_ids_lists": [torch.LongTensor(dna_ids) for dna_ids in dna_ids_lists],
            "dna_start_pos_lists": torch.LongTensor(dna_start_pos_lists)
        }

    def _collate_batch(self, samples: list):
        input_ids_list, dna_ids_lists, dna_start_pos_lists = [], [], []
        attention_masks_list = []
        for instance in samples:
            input_ids = torch.LongTensor(instance["input_ids"]) if isinstance(instance["input_ids"], list) else instance["input_ids"]
            attention_masks = torch.LongTensor(instance["attention_mask"])
            dna_ids_list = instance.get("dna_ids_lists", None)
            dna_start_pos_list = instance.get("dna_start_pos_lists", None)

            input_ids_list.append(input_ids) 
            attention_masks_list.append(attention_masks)
            dna_ids_lists.append(dna_ids_list)
            dna_start_pos_lists.append(dna_start_pos_list)

        if None in dna_ids_lists or None in dna_start_pos_lists:
            dna_ids_lists = None
            dna_start_pos_lists = None

        return {"input_ids": torch.stack(input_ids_list).to(device=self.device),
                "attention_mask": torch.stack(attention_masks_list).to(device=self.device),
                "dna_ids_lists": dna_ids_lists,
                "dna_start_pos_lists": dna_start_pos_lists}

    def run_batch(self):
        args = self.args
        ext = args.dataset_path.split('.')[-1]
        reader = {
            'csv': pd.read_csv,
            'xlsx': pd.read_excel,
            'jsonl': partial(pd.read_json, lines=True),
            'json': partial(pd.read_json, lines=False)
        }[ext]
        df = reader(args.dataset_path)
        results = []

        # 调用 process_sample 处理每一行
        processed = [self.process_sample(text) for text in df['input'].tolist()[:30]]

        # 分批生成
        for i in tqdm(range(0, len(processed), args.batch_size), desc="Batch"):
            batch_samples = processed[i : i+args.batch_size]
            batch = self._collate_batch(batch_samples)
            out_ids = self.model.generate(
                **batch,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            texts = self.text_tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            results.extend(texts)
            print(texts[0])
            print(texts[1])

        # 4) 把 llm 输出写回 df
        # df['llm_output'] = results
        df_small = df.iloc[:30].copy()
        df_small['llm_output'] = results

        # 5) 保存到新的 JSONL 文件
        # out_file = os.path.join(args.output_path, 'predictions_with_llm_1_2000.jsonl')
        # os.makedirs(args.output_path, exist_ok=True)
        # df_small.to_json(out_file, orient='records', lines=True, force_ascii=False)
        # print(f"Saved predictions with LLM output to {out_file}")


if __name__ == "__main__":
    args = parse_args()
    inferer = MultiModalInfer(args)
    inferer.run_batch()
