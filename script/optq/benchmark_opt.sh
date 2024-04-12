CUDA_VISIBLE_DEVICES=6 python opt.py \
    facebook/opt-125m \
    wikitext2 \
    --wbits 3 \
    --groupsize -1 \
		--benchmark 128 \
		--load 'dummy.pt' \