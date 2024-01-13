import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np



def get_quant_params(weight, bits = 4):
    maxq = torch.tensor(2 ** bits - 1)
    dev = weight.device
    N,K = weight.shape
    maxq = maxq.to(dev)
    shape = weight.shape
    tmp = torch.zeros(weight.shape[0], device=dev)
    xmin = torch.minimum(weight.min(1)[0], tmp)
    xmax = torch.maximum(weight.max(1)[0], tmp)
    
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1
    scale = (xmax - xmin) / maxq
    # zero = torch.round(-xmin / scale)
    zero = xmin

    return scale, zero, maxq


def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round((x - zero.unsqueeze(1).expand_as(x)) / scale.unsqueeze(1).expand_as(x)) , 0, maxq)
    # q = torch.clamp(torch.round(x / scale.unsqueeze(1).expand_as(x)) + zero.unsqueeze(1).expand_as(x), 0, maxq)

    return q.to(torch.int8)

def de_quantize(q, scale, zero):
    return scale.unsqueeze(1).expand_as(q) * q + zero.unsqueeze(1).expand_as(q)
    # return scale.unsqueeze(1).expand_as(q) * (q - zero.unsqueeze(1).expand_as(q))


def round(x, rounding='deterministic'):
    assert(rounding in ['deterministic', 'stochastic'])
    if rounding == 'stochastic':
        x_floor = x.floor()
        return x_floor + torch.bernoulli(x - x_floor)
    else:
        return x.round()

def get_shift_and_sign(x, rounding='deterministic'):
    sign = torch.sign(x)
    
    x_abs = torch.abs(x)
    if rounding == "floor":
        shift = torch.floor(torch.log(x_abs) / np.log(2))
    else:
        shift = round(torch.log(x_abs) / np.log(2), rounding)
    
    return shift, sign    

def round_power_of_2(x, rounding='deterministic', q_bias=None, scale=None):
    if q_bias is not None:
        q_bias = q_bias.unsqueeze(1).expand_as(x)
        x = x - q_bias
    if scale is not None:
        scale = scale.unsqueeze(1).expand_as(x)
        x = x / scale
    shift, sign = get_shift_and_sign(x, rounding)    
    x_rounded = (2.0 ** shift) * sign
    if scale is not None:
        x_rounded = x_rounded * scale
    if q_bias is not None:
        x_rounded = x_rounded + q_bias
    return x_rounded

def additive_power_of_2(x, nums = 2):
    shift, sign = get_shift_and_sign(x, rounding = "floor")
    x_rounded = (2.0 ** shift) * sign
    for _ in range(nums - 1):
        x = x - x_rounded
        shift, sign = get_shift_and_sign(x, rounding = "floor")
        x_rounded += (2.0 ** shift) * sign
    return x_rounded

def main():

    N = 128
    K = 256
    origin_bits = 5
    target_bits = 3
    for i in range(20):
        print("\n================================\n")

        weight = torch.randn(N, K, device = 'cuda')
        scale, zero, maxq = get_quant_params(weight, bits=target_bits)
        q = quantize(weight, scale, zero, maxq)
        print(torch.unique(q, return_counts=True))
        dq = de_quantize(q.to(torch.float), scale, zero)
        loss = F.mse_loss(weight, dq)
        print(f"linear quant loss({i}) : {loss}")

        dq_pot = round_power_of_2(weight)
        loss = F.mse_loss(weight, dq_pot)
        print(f"pot quant loss({i}) : {loss}")


        dp_apot = additive_power_of_2(weight, nums = 2)
        loss = F.mse_loss(weight, dp_apot)
        print(f"apot quant loss({i}) : {loss}")

if __name__ == "__main__":
    main()