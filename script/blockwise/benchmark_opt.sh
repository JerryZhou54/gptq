# CUDA_VISIBLE_DEVICES=6 python opt.py \
#     facebook/opt-125m \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize 96 \
# 		--benchmark 128 \
# 		--load 'dummy.pt' \

# CUDA_VISIBLE_DEVICES=6 python opt.py \
#     facebook/opt-350m \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize 128 \
# 		--lut_bench \
# 		--benchmark 128 \

# CUDA_VISIBLE_DEVICES=6 python opt.py \
#     facebook/opt-1.3b \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize 256 \
# 		--lut_bench \
# 		--benchmark 128 \

# CUDA_VISIBLE_DEVICES=6 python opt.py \
#     facebook/opt-6.7b \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize 512 \
# 		--lut_bench \
# 		--benchmark 128 \

# CUDA_VISIBLE_DEVICES=6 python opt.py \
#     facebook/opt-13b \
#     wikitext2 \
#     --wbits 3 \
#     --groupsize 640 \
# 		--lut_bench \
# 		--benchmark 128 \

CUDA_VISIBLE_DEVICES=6 python opt.py \
    facebook/opt-30b \
    wikitext2 \
    --wbits 3 \
    --groupsize 896 \
		--lut_bench \
		--benchmark 128 \