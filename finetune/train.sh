export CUDA_VISIBLE_DEVICES=0,2,4,5,6,7
export WANDB_PROJECT=bitllama-wikitext

python train.py \
    --dataset_name='wikitext' \
    --dataset_config_name='wikitext-2-raw-v1' \
    --model_name_or_path "facebook/opt-125m" \
    --num_train_epochs=10 \
    --block_size=2048 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --optim adafactor \
    --learning_rate=8e-4 \
    --torch_dtype bfloat16 \
    --bf16 \
    --output_dir='./llama-wikitext' \
    --do_train \
    --do_eval \
    --save_strategy='epoch' \
    --logging_strategy='steps' \
    --logging_first_step \
    --logging_steps=10 \
    --save_total_limit=1 \
    --run_name='llama-wikitext' \
    --overwrite_output_dir \
    --low_cpu_mem_usage