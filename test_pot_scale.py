import torch
import torch.nn as nn
from torch.autograd import Function

import numpy as np

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
    shift = round(torch.log(x_abs) / np.log(2), rounding)
    
    return shift, sign    

def round_power_of_2_function(x, rounding='deterministic'):
    shift, sign = get_shift_and_sign(x, rounding)    
    x_rounded = (2.0 ** shift) * sign
    return x_rounded


class RoundPowerOf2(Function):
    @staticmethod 
    def forward(ctx, input):
        return round_power_of_2_function(input)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output
        
def round_power_of_2(input):
    return RoundPowerOf2.apply(input)


def get_best_scale(input):
    
    critation = nn.MSELoss()
    scale = 1.0
    bias = 0.0

    quanted_input = round_power_of_2((input - bias) / scale)
    loss = critation(quanted_input * scale, input)
    print(f"loss: {loss}\n")
    print("input mean: ", torch.mean(input))
    print("input std: ", torch.std(input))
    print("quanted_input mean: ", torch.mean(quanted_input))
    print("quanted_input std: ", torch.std(quanted_input))


    for _ in range(5):
        scale = torch.std(input) / torch.std(quanted_input)
        bias = torch.mean(input) - torch.mean(quanted_input) * scale
        quanted_input = round_power_of_2((input - bias) / scale)
        loss = critation(quanted_input * scale + bias, input)

        print(f"loss: {loss}\n")
        print("\ninput mean: ", torch.mean(input))
        print("input std: ", torch.std(input))
        print("quanted_input mean: ", torch.mean(quanted_input))
        print("quanted_input std: ", torch.std(quanted_input))


    print(f"loss: {loss}\n")
    print(f"scale: {scale}\n")
    print(f"bias: {bias}\n")

    return scale, loss




input = torch.tensor([0.1, 0.3, 1.0, 2.0, 3.0, 4.0])
scale, loss = get_best_scale(input)
# print(f"scale: {scale}\n")
# print(f"loss: {loss}\n")