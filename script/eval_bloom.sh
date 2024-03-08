# CUDA_VISIBLE_DEVICES=6 python bloom.py \
#     bigscience/bloom-560m \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize -1 \
#     --lut_eval \
#     --bcq_round 5 


#     #     --nearest


# CUDA_VISIBLE_DEVICES=5 python bloom.py \
#     bigscience/bloom-1b1 \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize -1 \
#     --non_linear \
#     --hyperbits 6 \
#     --exploreBits 1 \
#     --exploreSplit 20

# CUDA_VISIBLE_DEVICES=4 python bloom.py \
#     bigscience/bloom-560m \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize -1 \
#     --columnwise \
#     --bcq_round 50 \
#     --apot_nums 2 \
#     --use_bst \

CUDA_VISIBLE_DEVICES=5 python bloom.py \
    bigscience/bloom-560m \
    wikitext2 \
    --wbits 3