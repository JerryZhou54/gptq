import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

model_path = '/data/hyou37/yipin/shiftaddllm/OPTQ/finetune/llama-wikitext/model.safetensors'

tensors = {}
with safe_open(model_path, framework="pt", device='cpu') as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

# print(tensors)
print(tensors.keys())