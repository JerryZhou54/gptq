CUDA_VISIBLE_DEVICES=4 python llama.py \
    /research/data/hyou37/llama/huggingface/llama-2-13b-chat \
    wikitext2 \
    --wbits 4 \
    --groupsize -1 