import math
import time
import os

from tqdm import tqdm
import torch
import torch.nn as nn
import transformers

from quant import *

from bcq_quant.quantizer import quantize as bcq_quantize
from bcq_quant.bcq_shift import quantize_shift
from plot_activation import plot_distribution2d

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False



class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.input = torch.mean(inp, 1)
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def analyse(self, percdamp=0.01):
        result = {
            "rowwise": {"w": {"max": None, "min": None, "mean": None, "std": None}, # analysis of weight
                        "wa" : { "max": None, "min": None, "mean": None, "std": None}, # analysis of weight * activation
                        "wh" : { "max": None, "min": None, "mean": None, "std": None}, # analysis of weight / hessian
            },
            "columnWise": {"w": {"max": None, "min": None, "mean": None, "std": None}, # analysis of weight
                        "wa" : { "max": None, "min": None, "mean": None, "std": None}, # analysis of weight * activation
                        "wh" : { "max": None, "min": None, "mean": None, "std": None}, # analysis of weight / hessian
            },
        }
        W = self.layer.weight.data.clone()
        W = W.float()
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)

        # analysis of weight
        result["rowwise"]["w"]["max"] = W.max(dim=1).values
        result["rowwise"]["w"]["min"] = W.min(dim=1).values
        result["rowwise"]["w"]["mean"] = W.mean(dim=1)
        result["rowwise"]["w"]["std"] = W.std(dim=1)
        result["columnWise"]["w"]["max"] = W.max(dim=0).values
        result["columnWise"]["w"]["min"] = W.min(dim=0).values
        result["columnWise"]["w"]["mean"] = W.mean(dim=0)
        result["columnWise"]["w"]["std"] = W.std(dim=0)

        # analysis of weight * activation
        weightAct = W * self.input.repeat(self.rows, 1) 
        result["rowwise"]["wa"]["max"] = weightAct.max(dim=1).values
        result["rowwise"]["wa"]["min"] = weightAct.min(dim=1).values
        result["rowwise"]["wa"]["mean"] = weightAct.mean(dim=1)
        result["rowwise"]["wa"]["std"] = weightAct.std(dim=1)
        result["columnWise"]["wa"]["max"] = weightAct.max(dim=0).values
        result["columnWise"]["wa"]["min"] = weightAct.min(dim=0).values
        result["columnWise"]["wa"]["mean"] = weightAct.mean(dim=0)
        result["columnWise"]["wa"]["std"] = weightAct.std(dim=0)


        weightH = W / torch.diag(H).repeat(self.rows, 1) 
        result["rowwise"]["wa"]["max"] = weightH.max(dim=1).values
        result["rowwise"]["wa"]["min"] = weightH.min(dim=1).values
        result["rowwise"]["wa"]["mean"] = weightH.mean(dim=1)
        result["rowwise"]["wa"]["std"] = weightH.std(dim=1)
        result["columnWise"]["wa"]["max"] = weightH.max(dim=0).values
        result["columnWise"]["wa"]["min"] = weightH.min(dim=0).values
        result["columnWise"]["wa"]["mean"] = weightH.mean(dim=0)
        result["columnWise"]["wa"]["std"] = weightH.std(dim=0)

        return result

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, 
        model_name = "opt", layer_name = "layer", lut_quant=False, non_linear_quant=False, columnwise=False,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # plot_distribution2d(W, file_path = f"./plot_activation/{model_name}_{layer_name}_origin.png")

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0


        # if not self.quantizer.ready():
        #     if non_linear_quant:
        #         self.quantizer.find_params(W, torch.diag(H))
        #     else:
        #         self.quantizer.find_params(W)

        # if lut_quant:
        #     print(self.quantizer.alpha[0:5,0,:])

        if static_groups and not lut_quant and not columnwise:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        if not self.quantizer.ready():
            if non_linear_quant:
                self.quantizer.find_params(W, self.input.float())
            elif not columnwise:
                self.quantizer.find_params(W)

        # if lut_quant:
        #     print(self.quantizer.alpha[0:5,0,:])
        # sensitive_block_idx = []
        # for _ in range(2):
        for i1 in tqdm(range(0, self.columns, blocksize), desc=layer_name, leave=False):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if lut_quant:
                    
                    if groupsize != -1:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        group = idx // groupsize
                    else:
                        group = 0
                    alpha = self.quantizer.alpha[:,group,:].unsqueeze(1)
                    q, BinaryWeight = bcq_quantize(w.unsqueeze(1), alpha, groupsize=-1)
                    q = q.flatten()
                elif non_linear_quant:
                    if groupsize != -1:
                        if not static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                        else:
                            idx = i1 + i
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // groupsize]
                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()

                elif columnwise:
                    # wf = torch.ones(w.unsqueeze(0).shape, dtype=w.dtype, device=w.device) / d 
                    # wf = None
                    # q, BinaryWeight, alpha, _, scale  = quantize_shift(w.unsqueeze(0),\
                    #         # qbits=self.quantizer.wbits if i1+i not in sensitive_block_idx else self.quantizer.wbits+1, 
                    #         qbits=self.quantizer.wbits, 
                    #         group_size=groupsize, rounds=self.quantizer.rounds, wf = wf, 
                    #         use_bst=self.quantizer.use_bst, apot_nums=self.quantizer.apot_nums)
                    # q = q.flatten()

                    wf = None
                    if i % 8 == 0:
                        w_8column = W1[:, i:i+8].flatten()
                        q, BinaryWeight, alpha, _, scale  = quantize_shift(w_8column.unsqueeze(0),
                    #            qbits=self.quantizer.wbits if i1+i not in sensitive_block_idx else self.quantizer.wbits+1,
                                qbits=self.quantizer.wbits,
                                group_size=groupsize * 8 if groupsize != -1 else -1, 
                                rounds=self.quantizer.rounds, wf = wf, 
                                use_bst=self.quantizer.use_bst, apot_nums=self.quantizer.apot_nums)
                    q, BinaryWeight = bcq_quantize(w.unsqueeze(0), alpha, groupsize=groupsize, use_bst=self.quantizer.use_bst)
                    q = q.flatten()

                else:
                    if groupsize != -1:
                        if not static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                        else:
                            idx = i1 + i
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // groupsize]

                    q = quantize(
                        w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

            # Q_block = Q.norm(dim=0).view(-1, 8).sum(dim=1)
            # _, idx = Q_block.topk(int(self.columns/8 * 0.05))
            # sensitive_block_idx = idx * 8
                    
            # Q_block = Q.norm(dim=0)
            # _, idx = Q_block.topk(int(self.columns * 0.05))
            # sensitive_block_idx = idx      

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        # save layername and error to file
        # with open(f"sensitivity/{model_name}.txt", "a+") as f:
        #     f.write(f"{layer_name}: {str(torch.sum(Losses).item())}\n")

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        # plot_distribution2d(Q, file_path = f"./plot_activation/{model_name}_{layer_name}_optq3bit.png")
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
