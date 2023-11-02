
# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-350m \
#     wikitext2 \
#     --benchmark 128 

# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-350m \
#     wikitext2 \
#     --wbits 3 \
#     --save weight/opt350m-3bit.pt


CUDA_VISIBLE_DEVICES=0 python opt.py \
    facebook/opt-350m \
    wikitext2 \
    --wbits 3 \
    --load weight/opt350m-3bit.pt \
    --benchmark 128 


# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-125m \
#     wikitext2 \
#     --benchmark 128 

# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-125m \
#     wikitext2 \
#     --wbits 3 \
#     --save weight/opt125m-3bit.pt


# CUDA_VISIBLE_DEVICES=0 python opt.py \
#     facebook/opt-125m \
#     wikitext2 \
#     --wbits 3 \
#     --load weight/opt125m-3bit.pt \
#     --benchmark 128 


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