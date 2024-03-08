
# CUDA_VISIBLE_DEVICES=4 python opt.py \
#     facebook/opt-125m \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize -1 \
#     --lut_eval \
#     --bcq_round 50 \
#     --apot_nums 3 \
#     --use_bst \

# CUDA_VISIBLE_DEVICES=6 python opt.py \
#     facebook/opt-13b \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize -1 \
#     --non_linear \
#     --hyperbits 5 \
#     --exploreBits 1 \
#     --exploreSplit 50


CUDA_VISIBLE_DEVICES=5 python opt.py \
    facebook/opt-30b \
    wikitext2 \
    --wbits 2 \
    --groupsize -1 \
    --columnwise \
    --bcq_round 50 \
    --apot_nums 3 \
    --use_bst \

# CUDA_VISIBLE_DEVICES=6 python opt.py \
#     facebook/opt-125m \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize -1 \
#     --bcq \
#     --bcq_round 50 \
#     --use_bst \