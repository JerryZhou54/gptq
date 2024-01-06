import torch
import torch.nn as nn
import numpy as np

from .bcq_shift import quantize_shift, find_B_torch

def quantize(x, alpha, groupsize = -1, scale = None, qbias = None):

    alpha.to(x.device)
    N, K = x.shape
    wbits = alpha.shape[-1]
    if groupsize == -1:
        groupsize = K
    x = x.reshape([N, K // groupsize, groupsize])
    w = x.clone()
    B = torch.zeros(N, K // groupsize, groupsize, wbits, device=x.device)

    if qbias is not None:
        qbias.to(qbias.device)
        qbias = qbias.unsqueeze(-1).expand_as(x)
        x = x - qbias
    if scale is not None:
        scale.to(x.device)
        scale = scale.unsqueeze(-1).expand_as(x)
        x = x / scale

    # B[:, :, :, 0] = torch.sign(w)
    # for i in range(1, wbits):
    #     w = w - B[:, :, :, i - 1] * alpha[:, :, i - 1].unsqueeze(-1).expand_as(w)
    #     B[:, :, :, i] = torch.sign(w)
    
    B = find_B_torch(x.reshape(-1, groupsize), alpha.reshape(-1, wbits))
    B = B.reshape([N, K // groupsize, groupsize, wbits])
    
    ret = torch.einsum('mngb,mnb->mng', (B, alpha))
    if scale is not None:
        ret = ret * scale
    if qbias is not None:
        ret = ret + qbias

    ret = ret.reshape([N, K])

    return ret, B


class BCQuantizer(nn.Module):

    def __init__(self, layer, groupsize=-1, wbits=3, rounds = 5):
        super(BCQuantizer, self).__init__()

        self.wbits = wbits
        self.groupsize = groupsize
        self.rounds = rounds

        W = layer.weight.data.clone()
        N, K = W.shape
        if groupsize == -1:
            num_group = 1
        else:
            if K % groupsize != 0:
                raise ValueError(r'K % groupsize != 0')
            num_group = K // groupsize

        self.register_buffer('alpha', torch.zeros(N, num_group, wbits))
        self.register_buffer('scale', torch.ones(N, num_group))
        self.register_buffer('qbias', torch.zeros(N, num_group))


    def find_params(self, x):
        # WARNING: assert x is linear weight
        if len(x.shape) != 2:
            raise ValueError(r'x should be linear weight')
        # self.ret, self.B, self.alpha, _, self.scale = \
        _, _, self.alpha, _, self.scale = \
            quantize_shift(x, qbits=self.wbits, rounds=self.rounds, group_size=self.groupsize)
        assert torch.all(torch.sort(self.alpha, dim=2, descending=True)[0] == self.alpha), "alpha should be in descending order, something wrong with 'quantize_shift'"

    def quantize(self, x):
        if not self.ready():
            self.find_params(x)
        return quantize(x, self.alpha, self.groupsize, self.scale, self.qbias)


    def ready(self):
        return torch.any(self.alpha != 0)


if __name__ == "__main__":
    layer = nn.Linear(128, 256)
    quantizer = BCQuantizer(layer, groupsize=-1, wbits=3, rounds=5)
    quantizer.find_params(layer.weight.data)
    ret, B = quantizer.quantize(layer.weight.data)
    print(B - quantizer.B)
    print(torch.norm(B - quantizer.B))
    assert torch.all(abs(B - quantizer.B)<1e-4)
    assert torch.all(abs(ret - quantizer.ret.cpu())<1e-4)
    