CUDA_VISIBLE_DEVICES=6 python opt.py \
    facebook/opt-125m \
    wikitext2 \
    --wbits 3 \
    --groupsize -1 \
    --bcq \
    --bcq_round 50 \
    --use_bst \