import time

import torch
import torch.nn as nn

import lutgemm_cuda

DEV = torch.device('cuda:0')

# M = 12288
# N = 12288

DTYPE = torch.float

input = torch.randn((1, 12288), device=DEV, dtype=DTYPE)
output = torch.zeros((1, 12288), device=DEV, dtype=DTYPE)

weight = torch.randint(-1000000000, 1000000000, (12288, 96, 128 // 32 * 4), device=DEV, dtype=torch.int)
alpha = torch.randn(12288, 96, 4, device=DEV, dtype=DTYPE)

wbit = 4

COUNT = 100000
tick = time.time()
for _ in range(COUNT):
    lutgemm_cuda.lutMatmul(weight, alpha, input, output, wbit)
    # torch.cuda.synchronize()
print(output)
print('LUT-Gemm 4bit:', (time.time() - tick) / COUNT)
