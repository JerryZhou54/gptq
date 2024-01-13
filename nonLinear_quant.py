import numpy as np
import torch
import torch.nn as nn
from itertools import combinations

from tqdm import tqdm
import torch.nn.functional as F

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
    zero = torch.round(xmin / scale) * scale

    return scale, zero, maxq


def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round((x - zero.unsqueeze(1).expand_as(x)) / scale.unsqueeze(1).expand_as(x)) , 0, maxq)
    # q = torch.clamp(torch.round(x / scale.unsqueeze(1).expand_as(x)) + zero.unsqueeze(1).expand_as(x), 0, maxq)

    return q

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
    try:
        array = torch.tensor(array, device = dev)
        diff = torch.abs(weight - array) # 2**n times the parameter, cause a lot of memory usage
        nearest_indices = diff.argmin(dim=1)
        nearest_values = array[nearest_indices]
    except: # out of memory
        # weight = weight.to("cpu")
        # array = torch.tensor(array, device = "cpu")
        array = torch.tensor(array, device = dev)
        min_diff = torch.abs(weight - array[0])
        min_indices = torch.zeros_like(weight).to(torch.int)
        for i, number in enumerate(array[1:], 1):
            diff = torch.abs(weight - number)
            mask = diff < min_diff
            min_diff[mask] = diff[mask]
            min_indices[mask] = int(i)
        nearest_values = array[min_indices]
        nearest_values = nearest_values.to(dev)
    return nearest_values.reshape(shape)

class NonLinearQuantizer(nn.Module):

    def __init__(self, layer, wbits=3, hyperbits = 5, exploreBits = 1, exploreSplit = 20):
        super(NonLinearQuantizer, self).__init__()

        self.wbits = wbits
        self.hyperbits = hyperbits
        self.exploreBits = exploreBits
        self.exploreSplit = exploreSplit

        W = layer.weight.data.clone()
        N, K = W.shape

        self.register_buffer('maxq', torch.tensor(N))
        self.register_buffer('scale', torch.zeros(N))
        self.register_buffer('zero', torch.zeros(N))
        self.choice_bits = None


    def find_params(self, x, input=None):
        # WARNING: assert x is linear weight
        if len(x.shape) != 2:
            raise ValueError(r'x should be linear weight')
        
        all_choice = get_all_choice(origin_bits = self.hyperbits, target_bits = self.wbits)
        best = {
            'loss': float('inf'),
            'choice': None,
            'scale': None,
            'zero': None,
        }

        tmp = torch.zeros(x.shape[0], device=x.device)
        maxq = torch.tensor(2 ** self.hyperbits - 1, device=x.device)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        scale = (xmax - xmin) / maxq
        zero = torch.round(xmin / scale) * scale
        scale_down = (xmax - xmin) / (2 ** (self.hyperbits+self.exploreBits) - 1)
        scale_up = (xmax - xmin) / (2 ** (self.hyperbits-self.exploreBits) - 1)
        step = (scale_up - scale_down) / self.exploreSplit
        # print(f"scale_down : {scale_down[:3]}, scale_up : {scale_up[:3]}, step : {step[:3]}, base_scale : {scale[:3]}")
        for each_choice in tqdm(all_choice, desc = 'find best bit', leave = False):
            for i in range(self.exploreSplit + 1):
                scale = scale_down + step * i
                # zero = torch.round(xmin / scale) * scale

                q = quantize(x, scale, zero, maxq)
                q = nearest_value(q, each_choice)
                dq = de_quantize(q, scale, zero)
                if input is None:
                    loss = F.mse_loss(x, dq)
                else:
                    loss = F.mse_loss(x.matmul(input), dq.matmul(input))
                # print(f"non-linear quant loss({i}) : {loss}")
                if loss < best["loss"]:
                    best["loss"] = loss
                    best["choice"] = each_choice
                    best["scale"] = scale
                    best["zero"] = zero
        
        self.scale = best["scale"]
        self.zero = best["zero"]
        self.maxq = maxq
        self.choice_bits = best["choice"]
        # print(f"non-linear quant loss : {best['loss']}, best_choice : {best['choice']}, scale : {best['scale'][:3]}")
        torch.cuda.empty_cache()

    def quantize(self, x):
        if self.ready():
            q = quantize(x, self.scale, self.zero, self.maxq)
            q = nearest_value(q, self.choice_bits)
            dq = de_quantize(q.to(torch.float), self.scale, self.zero)
            return dq
        else:
            raise ValueError('Quantizer not ready.')

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

