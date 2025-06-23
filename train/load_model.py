import os
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.utils import is_liger_kernel_available
if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM as AutoModelForCausalLM

from model import *
from common.lora_modules import *
from common.registry import registry
from common.utils.params_manager import set_up_multimodal_config
from common.utils import (
    load_ckpt_for_train,
    print_rank_0, read_config, load_ckpt,
    dict_to_dataclass, set_default_tensor_type, STR_DTYPE_TO_TORCH_DTYPE)

def load_local_model(args):
    return_dataset_kwargs = {}
    # Train with local defined model.
    print_rank_0(f'--->Using tokenizer: {args.tokenizer_name} with path: {args.tokenizer_path}', args.global_rank)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    config_type = '_'.join([args.model_name, args.variant])
    model_config = registry.get_model_config_class(config_type)(args.ckpt_path, args.multimodal_model_ckpt_path)
    set_up_multimodal_config(model_config, args)
    print_rank_0(f'--->Using model config: {config_type}', args.global_rank)
    # 需要修改，Qwen与LLaMA, Qwen就可以使用
    # model_config.vocab_size = tokenizer.n_words
    # Load model in default dtype to avoid OOM (cpu memory).
    # model_config.multimodal_model_config.atten_type = "flash_atten"
    # model_config.multimodal_model_config.add_pooling_layer = False
    with set_default_tensor_type(args.default_dtype):
        model = registry.get_model_class(args.model_name)(model_config)

    print_rank_0(f'--->Using model: {args.model_name}, and loading its trainning variant', args.global_rank)

    # Load checkpoint if checkpoint path is provieded.
    if args.ckpt_path is not None and args.from_pretrained:
        model.model.load_state_dict(Qwen3ForCausalLM.from_pretrained(args.ckpt_path, trust_remote_code=True).state_dict())
        print_rank_0(f'--->Using pretrained checkpoint at {args.ckpt_path}', args.global_rank)
        # model_config.optimizer_sd = optimizer_sd
        # model_config.lr_scheduler_sd = lr_scheduler_sd

        if args.multimodal and hasattr(model, 'bio_model') and args.multimodal_model_ckpt_path:
            return_dataset_kwargs['multimodal_k_tokens'] = args.multimodal_k_tokens
            model.bio_model.load_state_dict(AutoModel.from_pretrained(args.multimodal_model_ckpt_path, config=model_config.multimodal_model_config, trust_remote_code=True).state_dict(), strict=False)
            print_rank_0(f'--->Using pretrained multimodal model checkpoint at {args.multimodal_model_ckpt_path}', args.global_rank)
            multimodal_tokenizer = AutoTokenizer.from_pretrained(args.multimodal_model_ckpt_path, trust_remote_code=True)
            return_dataset_kwargs['multimodal_tokenizer'] = multimodal_tokenizer
    else:
        print_rank_0('--->Not using pretrained checkpoint to start traning.', args.global_rank)


    # Load config from model config to argument parser namespace. 🌟这里需要注意，替换到下面
    # args.head_dim = model_config.head_dim
    # args.head_num = model_config.n_heads
    # args.hidden_size = model_config.dim
    # args.num_layers = model_config.n_layers
    # args.rope_theta = model_config.rope_theta if args.rope_theta is None else args.rope_theta
    # args.pad_id = tokenizer.pad_id

    # for qwen
    args.head_dim = model_config.hidden_size // model_config.num_attention_heads
    args.head_num = model_config.num_attention_heads
    args.hidden_size = model_config.hidden_size
    args.num_layers = model_config.num_hidden_layers
    args.rope_theta = model_config.rope_theta if args.rope_theta is None else model_config.rope_theta
    args.pad_id = tokenizer.pad_token_id

    # Convert dtype to avoid inconsistency between default dtype and checkpoint dtype.
    torch.cuda.empty_cache()
    gc.collect()
    model.to(STR_DTYPE_TO_TORCH_DTYPE[args.default_dtype]).to(args.device)
    return model, tokenizer, model_config, return_dataset_kwargs

def load_model(args):
    """
    Load model from args.
    If `args.huggingface`, then the model and tokenizer will be loaded from transformers.
    Else the model and tokenizer will be load from MyTransformers.

    Note that huggingface models do not support pipeline parallelism and sequence parallelism for now.

    It is possisble to support sequence parallelism by replace the Attention class of huggingface models 
    using a monkey patch.
    """
    return load_local_model(args)