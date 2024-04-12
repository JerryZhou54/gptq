CUDA_VISIBLE_DEVICES=6 python opt.py \
    facebook/opt-30b \
    wikitext2 \
    --wbits 3 \
    --groupsize -1 \