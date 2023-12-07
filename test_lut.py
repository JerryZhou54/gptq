import torch 
import lutgemm
M, N, K = 1, 768, 768
A_GROUP_SIZE = 128
num_groups = K//A_GROUP_SIZE
NUM_BITS = 3
K_new = K * 3 // 32

device = 1

def get_tensors(M, N, K, num_groups, NUM_BITS):
    weight = torch.zeros(K, N, dtype=torch.float, device="cpu")
    input = torch.zeros(M, K, dtype=torch.float, device="cpu")
    output = torch.zeros(M, N, dtype=torch.half, device="cpu")
    alpha = torch.zeros(num_groups, NUM_BITS, N, dtype=torch.float, device="cpu")
    q_bias = torch.zeros(num_groups, N, dtype=torch.float, device="cpu")
    scale = torch.zeros(N, dtype=torch.float, device="cpu")
    bias = torch.zeros(N, dtype=torch.float, device="cpu")
    weight_int3 = torch.zeros(K * 3 // 32, N, dtype=torch.int32, device="cpu")
    qW = torch.zeros(K, NUM_BITS, N, dtype=torch.float, device="cpu")
    bW = torch.zeros(K//32, NUM_BITS, N, dtype=torch.int32, device="cpu") #[K/32][NUM_BITS][N];
    #bWeight = torch.empty(kSize * mSize * nb / 32, NUM_BITS, N, dtype=torch.float, device="cpu")
    return weight, input, output, alpha, q_bias, scale, bias, weight_int3, qW, bW

weight, input, output, alpha, q_bias, scale, bias, weight_int3, qW, bW  = get_tensors(M, N, K, num_groups, NUM_BITS)

lutgemm.makeRandomInput(input, M, K)
lutgemm.makeRandomBias(bias, N)
lutgemm.makeRandomAlpha(alpha, q_bias, num_groups, NUM_BITS, N)
lutgemm.makeRandomScale(scale, N)
lutgemm.makeRandomWeight_int3(weight_int3, N, K_new)
lutgemm.makeRandomWeight(qW, bW, N, NUM_BITS, K)
print("original: ", bW.size(), alpha.size(), q_bias.size(), input.size(), bW.dtype, alpha.dtype, q_bias.dtype, input.dtype)
bWeight, p_q_bias, p_alpha = lutgemm.parsing(bW,alpha.view(-1), K, N, NUM_BITS, False, num_groups, q_bias.view(-1), device)

mSize, kSize = N, K
output = output.to(device)
input = input.to(device).half()
#print("size: ", output.size(), bWeight.size(), p_alpha.size(), input.size(), p_q_bias.size(), mSize, kSize, NUM_BITS, num_groups)
lutgemm.lutgemm_compute(output, bWeight, p_alpha, p_q_bias, input, mSize, kSize, NUM_BITS, num_groups)
print(input.mean(), bWeight.size(), p_q_bias.size(), p_alpha.size(), bWeight.dtype, p_q_bias.dtype, p_alpha.dtype)
print(output, output.max(), output.mean())
print(output.shape, output.dtype)

import time
COUNT = 10000
# warmup
for _ in range(COUNT):
    lutgemm.lutgemm_compute(output, bWeight, p_alpha, p_q_bias, input, mSize, kSize, NUM_BITS, num_groups)
    torch.cuda.synchronize()

tick = time.time()
for _ in range(COUNT):
    lutgemm.lutgemm_compute(output, bWeight, p_alpha, p_q_bias, input, mSize, kSize, NUM_BITS, num_groups)
    torch.cuda.synchronize()
print('lut-gemm:', (time.time() - tick) / COUNT)