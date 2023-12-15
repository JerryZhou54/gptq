
# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-13b \
#     wikitext2 \
#     --wbits 3 \
#     --linearmix
    # --linearmix \
    # --layermix
    # --benchmark 128 



CUDA_VISIBLE_DEVICES=1 python opt.py \
    facebook/opt-2.7b \
    wikitext2 \
    --wbits 3 \
    --groupsize 256 \
    # --benchmark 128 \
    # --lut_bench


# ====================================================================================================

# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-6.7b \
#     wikitext2 \
#     --benchmark 128 

# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-6.7b \
#     wikitext2 \
#     --wbits 3 \
#     --load weight/opt6.7b-3bit.pt \
#     --benchmark 128 


# ====================================================================================================

# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-1.3b \
#     wikitext2 \
#     --benchmark 128 

# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-1.3b \
#     wikitext2 \
#     --wbits 3 \
#     --save weight/opt1.3b-3bit.pt


# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-1.3b \
#     wikitext2 \
#     --wbits 3 \
#     --load weight/opt1.3b-3bit.pt \
#     --benchmark 128 


# ====================================================================================================

# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-13b \
#     wikitext2 \
#     --benchmark 128 

# CUDA_VISIBLE_DEVICES=0,1,2,4,5 python opt.py \
#     facebook/opt-30b \
#     wikitext2 \
#     --wbits 2 \
    # --save weight/opt30b-3bit.pt

# CUDA_VISIBLE_DEVICES=0,1,2,4,5 python opt.py \
#     facebook/opt-13b \
#     wikitext2 \
#     --wbits 3 \
#     --save weight/opt13b-3bit.pt

# CUDA_VISIBLE_DEVICES=0,1,2,4,5 python opt.py \
#     facebook/opt-30b \
#     wikitext2 \
#     --wbits 3 \
#     --load weight/opt30b-3bit.pt \
#     --benchmark 128 