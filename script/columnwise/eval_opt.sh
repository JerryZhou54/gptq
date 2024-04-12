CUDA_VISIBLE_DEVICES=7 python opt.py \
    facebook/opt-13b \
    wikitext2 \
    --wbits 3 \
    --groupsize -1 \
    --columnwise \
    --bcq_round 10 \
    --apot_nums 3 \
    --use_bst