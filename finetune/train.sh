export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=bitllama-wikitext

    # --dataset_name='wikitext' \
    # --dataset_config_name='wikitext-2-raw-v1' \

python train.py \
    --dataset_name="tatsu-lab/alpaca" \
    --model_name_or_path "facebook/opt-125m" \
    --num_train_epochs=100 \
    --block_size=2048 \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 \
    --torch_dtype bfloat16 \
    --bf16 \
    --output_dir='./weights/opt125m-qlora' \
    --do_train \
    --do_eval \
    --save_strategy='epoch' \
    --logging_strategy='steps' \
    --logging_first_step \
    --logging_steps=20 \
    --save_total_limit=1 \
    --run_name='opt-alpaca' \
    --overwrite_output_dir \
    --low_cpu_mem_usage \
    --bits 4