# CUDA_VISIBLE_DEVICES=2 python gemma.py \
#     /research/data/hyou37/gemma/huggingface/gemma-7b \
#     wikitext2 \
#     --groupsize -1 

CUDA_VISIBLE_DEVICES=4 python gemma.py \
    /research/data/hyou37/gemma/huggingface/gemma-7b \
    wikitext2 \
    --wbits 4 \
    --groupsize -1 


# CUDA_VISIBLE_DEVICES=1 python opt.py \
#     facebook/opt-125m \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize -1 \
#     --columnwise \
#     --bcq_round 50 \
#     --apot_nums 2 \
#     --use_bst \

# CUDA_VISIBLE_DEVICES=6 python opt.py \
#     facebook/opt-125m \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize -1 \
#     --bcq \
#     --bcq_round 50 \
#     --use_bst \