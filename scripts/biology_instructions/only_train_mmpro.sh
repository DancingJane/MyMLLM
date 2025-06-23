export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

base_options="--train-dataset-name chat_multi_omics \
--eval-dataset-name biology_instructions \
--model-name qwen3_with_bert \
--tokenizer-name qwen3 \
--output-path /fs-computility/ai4agr/lijinzhe/res_data_model/0620_qwen3 \
--tokenizer-path /fs-computility/ai4agr/lijinzhe/basemodel/Qwen3-8B \
--ckpt-path /fs-computility/ai4agr/lijinzhe/basemodel/Qwen3-8B \
--tb-log-dir your_tensorboard_log_path \
--dataset-class-name iterable_multimodal_dna_dataset \
"

# enable_list="multimodal norm"
enable_list="multimodal"

options="$base_options \
    --multimodal \
    --multimodal-k-tokens 64 \
    --multimodal-model-ckpt-path /fs-computility/ai4agr/lijinzhe/basemodel/DNABERT-2-117M \
    --multimodal-tokenizer-path /fs-computility/ai4agr/lijinzhe/basemodel/DNABERT-2-117M \
    --multimodal-tokenizer-name dnabert2 \
    --experiment-name multimodal_qwen3_only_mmpro_exp_2 \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode sft \
    --prompt-path /fs-computility/ai4agr/lijinzhe/code/MyMLLM/prompt_config.json \
    --from-pretrained \
    --epochs 1 \
    --batch-size-per-gpu 4 \
    --eval-batch-size-per-gpu 4 \
    --eval-interval 10 \
    --save-interval 1000000 \
    --bf16 \
    --variant large \
    --device cuda \
    --max-len 1024 \
    --max-src-len 1024 \
    --eval-max-len 1024 \
    --eval-max-src-len 1024 \
    --seed 42 \
    --zero-stage 2 \
    --lr 1e-5 \
    --lr-decay-ratio 0.1 \
    --warmup 0.03 \
    --auto-warmup-steps 50 \
    --auto-warmup-rate 0.05 \
    --atten-type flash_atten \
    --wandb \
    --wandb-project BioMLLM_wandb \
    --wandb-dir BioMLLM_wandb \
    --wandb-cache-dir /fs-computility/ai4agr/lijinzhe/code/BioMLLM/scripts/biology_instructions/wandb \
    --diy-optimizer \
    --save-trainable \
    --enable-list $enable_list \
    "

deepspeed --include localhost:0,1,2,3,4,5,6,7 \
/fs-computility/ai4agr/lijinzhe/code/BioMLLM/train/u_train.py \
$options

