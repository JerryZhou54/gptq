CUDA_VISIBLE_DEVICES=0 python opt.py \
    facebook/opt-125m \
    wikitext2 \
    --wbits 3 \
    --groupsize 96 \
    --columnwise \
    --bcq_round 50 \
    --apot_nums 3 \
    --block_quant \
    --use_bst

# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-350m \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize 128 \
#     --columnwise \
#     --bcq_round 50 \
#     --apot_nums 3 \
#     --block_quant \
#     --use_bst

# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-1.3b \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize 256 \
#     --columnwise \
#     --bcq_round 50 \
#     --apot_nums 3 \
#     --block_quant \
#     --use_bst

# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-6.7b \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize 512 \
#     --columnwise \
#     --bcq_round 50 \
#     --apot_nums 3 \
#     --block_quant \
#     --use_bst

# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-13b \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize 640 \
#     --columnwise \
#     --bcq_round 50 \
#     --apot_nums 3 \
#     --block_quant \
#     --use_bst

# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-30b \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize 896 \
#     --columnwise \
#     --bcq_round 50 \
#     --apot_nums 3 \
#     --block_quant \
#     --use_bst