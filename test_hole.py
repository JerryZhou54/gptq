import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from itertools import combinations

from tqdm import tqdm

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


def get_all_choice(origin_bits = 4, target_bits = 3):
    all_alpha =[2**i for i in range(origin_bits-2, -2, -1)]
    all_choice = []
    for each_alpha_choice in combinations(all_alpha, target_bits):
        choice = [(2 ** origin_bits - 1) / 2]
        number_temp = []
        for alpha in each_alpha_choice:
            for each_number in choice:
                number_temp.append(each_number+alpha)
                number_temp.append(each_number-alpha)
            choice = number_temp
            number_temp = []
        all_choice.append(choice)
    return all_choice

def nearest_value(weight, array):
    dev = weight.device
    shape = weight.shape
    weight = weight.view(-1, 1)
    array = torch.tensor(array, device = dev)
    min_diff = torch.abs(weight - array[0])
    min_indices = torch.zeros_like(weight).to(torch.int)
    diff = torch.abs(weight.view(-1, 1) - array)
    for i, number in enumerate(array[1:], 1):
        diff = torch.abs(weight - number)
        mask = diff < min_diff
        min_diff[mask] = diff[mask]
        min_indices[mask] = int(i)
    nearest_values = array[min_indices]
    return nearest_values.reshape(shape)
        

def main():

    N = 128
    K = 256
    origin_bits = 5
    target_bits = 3
    for i in range(1):
        weight = torch.randn(N, K, device = 'cuda')
        scale, zero, maxq = get_quant_params(weight, bits=target_bits)
        q = quantize(weight, scale, zero, maxq)
        print(torch.unique(q, return_counts=True))
        dq = de_quantize(q.to(torch.float), scale, zero)
        loss = F.mse_loss(weight, dq)
        print(f"linear quant loss({i}) : {loss}")

        print(" -> non-linear quant start")
        all_choice = get_all_choice(origin_bits = origin_bits, target_bits = target_bits)
        best_choice = None
        best_loss = float('inf')
        for each_choice in tqdm(all_choice):
            scale, zero, maxq = get_quant_params(weight, bits=origin_bits)
            q = quantize(weight, scale, zero, maxq)
            q = nearest_value(q, each_choice)
            dq = de_quantize(q.to(torch.float), scale, zero)
            loss = F.mse_loss(weight, dq)
            # print(f"non-linear quant loss({i}) : {loss}")
            if loss < best_loss:
                best_loss = loss
                best_choice = each_choice
        print(f"non-linear quant loss({i}) : {best_loss}, best_choice : {best_choice}")

if __name__ == "__main__":
    # main()
    tensor = torch.tensor([1, 2, 3],device='cuda', dtype=torch.float)
    array = [0, 2, 4, 6]
    print(nearest_value(tensor, array))