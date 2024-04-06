CUDA_VISIBLE_DEVICES=7 python opt.py \
    facebook/opt-125m \
    wikitext2 \
    --wbits 3 \
    --groupsize -1 \
    --columnwise \
    --bcq_round 50 \
    --apot_nums 3 \
    --use_bst