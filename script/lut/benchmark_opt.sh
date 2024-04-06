CUDA_VISIBLE_DEVICES=6 python opt.py \
    facebook/opt-125m \
    wikitext2 \
    --wbits 3 \
    --groupsize -1 \
		--lut_bench \
		--benchmark 128 \