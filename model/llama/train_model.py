import torch
from torch.utils.checkpoint import checkpoint

from common.utils import parallel_states as parallel_states
from common.registry import registry
from model.base_model import BaseModel
from model.llama.model import LlamaGenerate, precompute_freqs_cis
from common.utils.torch_hooks import hook_backward_norm_fn, save_grad, ParameterUpdateHook

@registry.register_train_model(["llama", "llama1", "llama2", "llama3"])
class LLaMaTrainModel(BaseModel):
    """
    Trainer class for llama, responsible for handling input and output during training.
    """
    def __init__(self, model:LlamaGenerate, args):
        """
        Initializes basic attributes for the trainer class and precomputes fixed values.

        param model: llama model with pretrained weight.
        param args: Arguments from argument parser.
        """
        super().__init__(args)
        self.layers = model.model.layers
        self.tok_embeddings = model.model.tok_embeddings
        self.output = model.model.output
        self.norm = model.model.norm
        if 'flash' in args.atten_type:
            self.attention_mask = None
        else:
            self.attention_mask = LLaMaTrainModel.get_masks(args.max_len)
        self.freqs_cis = precompute_freqs_cis(args.head_dim,
                                            args.max_len,
                                            theta=args.rope_theta,
                                            train_pi=args.train_pi,
                                            train_pipeline=False)
        
    def forward(self, **kwargs):
        return super().forward(**kwargs)
    
    def embedding(self, input_ids):
        hidden_states = self.tok_embeddings(input_ids)
        if self.attention_mask is not None:
            attention_mask = self.attention_mask.to(hidden_states.device).to(hidden_states.dtype)
        else:
            attention_mask = self.attention_mask
        return hidden_states, attention_mask
    
    def model_forward(self, logits, labels, freqs_cis, attention_mask):
        # Using activation checkpoint to reduce memory consumption or not.
        for i in range(self.args.num_layers):
            if self.args.activation_checkpoint:
                logits = checkpoint(self.layers[i], 
                                    logits, 
                                    0, 
                                    freqs_cis, 
                                    attention_mask, 
                                    self.args.atten_type,
                                    use_reentrant=False)
            else:
                logits = self.layers[i](x=logits, 
                                        start_pos=0, 
                                        freqs_cis=freqs_cis, 
                                        mask=attention_mask, 
                                        atten_type=self.args.atten_type)
        logits = self.norm(logits)
        if self.fuse_linear_loss:
            loss = self.compute_loss(logits, labels, self.output.weight)
            logits = None
        else:
            logits = self.output(logits)
            loss = self.compute_loss(logits, labels)
        return loss, logits
    
    
    @staticmethod
    def get_masks(seqlen, device='cpu', dtype=torch.float, start_pos=0):
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=device),mask]).to(dtype)
            return mask


@registry.register_train_model(["llama1_with_hyena", "llama2_with_hyena", "llama3_with_hyena",
                                "llama1_with_bert", "llama2_with_bert", "llama3_with_bert", "llama3_with_nt"])
class MultimodalLlamaTrainModel(LLaMaTrainModel):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.encode_fp32 = args.multimodal_encode_fp32
        self.multimodal_model = model.multimodal_model
        self.multimodal_projector = model.multimodal_projector
        # self.layers[0].register_full_backward_hook(hook_backward_fn)
        self.multimodal_projector.register_full_backward_hook(hook_backward_norm_fn)
        # self.multimodal_model.register_full_backward_hook(hook_backward_norm_fn)
        # hook = ParameterUpdateHook('multimodal_projector', self.multimodal_projector.weight, print_every=10)
        # self.multimodal_projector.weight.register_hook(hook)

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]          # [batch_size, seq_len]
        dna_ids = kwargs["dna_ids"]             # [batch_size, 2, max_dna_len] 
        labels = kwargs["labels"]               # [batch_size, seq_len]
        before_dna = kwargs["before_dna"]       # [batch_size, 2] (第二个位置无效时为-1)
        
        # 1. 文本嵌入
        hidden_states = self.tok_embeddings(input_ids)  # [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 2. 处理DNA序列（两个片段）
        if self.encode_fp32:
            self.multimodal_model.to(torch.float32)
        
        # 处理每个DNA片段 (维度重组为 [batch_size*2, max_dna_len])
        dna_ids_flat = dna_ids.view(-1, dna_ids.size(-1))  # [batch_size*2, max_dna_len]
        dna_hidden_flat = self.encoder_forward(dna_ids_flat, hidden_states.dtype)  # [batch_size*2, max_dna_len, hidden_dim]
        dna_hidden_states = dna_hidden_flat.view(batch_size, 2, -1, hidden_dim)    # [batch_size, 2, dna_len, hidden_dim]
        
        if self.encode_fp32:
            dna_hidden_states = dna_hidden_states.to(input_ids.dtype)
        
        # 3. 逐个处理DNA片段
        for dna_idx in range(2):
            # 获取当前片段的起始位置（无效片段跳过）
            curr_before_dna = before_dna[:, dna_idx]  # [batch_size,]
            valid_mask = (curr_before_dna >= 0)       # 标记有效片段
            
            if not valid_mask.any():
                continue  # 所有样本的该片段都无效
                
            # 处理有效样本的当前DNA片段
            curr_dna_hidden = dna_hidden_states[valid_mask, dna_idx]  # [valid_count, dna_len, hidden_dim]
            curr_dna_len = curr_dna_hidden.shape[1]
            
            # 计算替换位置
            start_pos = curr_before_dna[valid_mask]   # [valid_count,]
            end_pos = start_pos + curr_dna_len
            
            # 创建掩码（仅对有效样本）
            valid_hidden = hidden_states[valid_mask]  # [valid_count, seq_len, hidden_dim]
            index = torch.arange(seq_len, device=valid_hidden.device).expand(valid_hidden.size(0), -1)  # [valid_count, seq_len]
            
            mask = (index >= start_pos.unsqueeze(1)) & (index < end_pos.unsqueeze(1))
            mask = mask.unsqueeze(-1).expand(-1, -1, hidden_dim)  # [valid_count, seq_len, hidden_dim]
            
            # 执行替换
            valid_hidden = valid_hidden.masked_scatter(
                mask, 
                curr_dna_hidden.reshape(-1, curr_dna_len * hidden_dim)
            )
            hidden_states[valid_mask] = valid_hidden
    
        # 4. 后续处理
        hidden_states, labels, freqs_cis = self.cut_sequence(hidden_states, labels)
        attention_mask = self.attention_mask.to(hidden_states.device) if self.attention_mask else None
        loss, _ = self.model_forward(hidden_states, labels, freqs_cis, attention_mask)
        
        return loss, {}
    
    def encoder_forward(self, dna_ids, origin_dtype):
        dna_hidden_states = self.multimodal_model(dna_ids)
        dna_hidden_states = self.multimodal_projector(dna_hidden_states)
        # dna_hidden_states.register_hook(save_grad('dna'))
        return dna_hidden_states