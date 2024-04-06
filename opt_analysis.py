import time
import math
import bisect

import torch
import torch.nn as nn
import numpy as np
import json

from gptq import *
from modelutils import *
from quant import *

from bcq_quant.quant_model_bcq import quant_model
# from lut_gemm.quant import load_lut
from bcq_quant.quantizer import BCQuantizer
from nonLinear_quant import NonLinearQuantizer
from plot_activation import plot_distribution

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')
    print('Analysing ...')

    analysis = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)

            result = gptq[name].analyse(args.percdamp)
            analysis['model.decoder.layers.%d.%s' % (i, name)] = result
            gptq[name].free()
        
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return analysis

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )

    parser.add_argument(
        '--load_weights', type=str, default=None,
        help='weights file to load. None for no weights.'
    )
    parser.add_argument(
        '--load_safetensor', action='store_true', default=True,
        help='Whether to load from a safetensor file.'
    )

    parser.add_argument(
        '--load_analyse_result', action='store_true', default=False,
        help='Whether to load from a safetensor file.'
    )
    parser.add_argument(
        '--anaylse_method', type=str, default="w", choices=["w","wa","wh","wn"],
        help='analysis sensitive on with value.'
    )
    parser.add_argument(
        '--mix_bits', type=str, default="0.0,0.8,0.2",
        help='The proportion of each bits. index=0 for 1bits quant, index=1 for 2bits quant. as default, 80% 2bit and 20% 3bit'
    )
    args = parser.parse_args()

    args.mix_bits = str(args.mix_bits).split(",")
    args.mix_bits = [float(each) for each in args.mix_bits]
    assert sum(args.mix_bits) == 1, "The sum of the ratios of the mixed bits should be 1"
    # get cumsum
    args.mix_bits = np.cumsum(np.array(args.mix_bits)).tolist()
    print(args.mix_bits)

    # Metrics are counted from weights
    if not args.load_analyse_result:
        model = get_opt(args.model)
        model.eval()
        print(model)

        if args.load_weights is not None:
            if args.load_safetensor:
                from safetensors import safe_open
                ckpt_weights = {}
                with safe_open(args.load_weights, framework="pt", device='cpu') as f:
                    for k in f.keys():
                        ckpt_weights[k] = f.get_tensor(k)
                model.load_state_dict(ckpt_weights, strict=False)
            else:
                model.load_state_dict(torch.load(args.load_weights), strict=True)

        dataloader, testloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
        )

        analysis_result = opt_sequential(model, dataloader, DEV)
        model_name = str(args.model).split("/")[-1]
        torch.save(analysis_result, f"./sensitivity/zeroshot/{model_name}.pth")
    else:
        model_name = str(args.model).split("/")[-1]
        analysis_result = torch.load(f"./sensitivity/zeroshot/{model_name}.pth")

    # Set up the quantization scheme for each layer
    model_quant_config = {}
    weight_score = []
    layers = []
    for each in analysis_result.keys():
        if "fc" in each:
            layers.extend(each for _ in range(4))
        else:
            layers.append(each)
    

    for each in layers:
        layer_quant_config = {
            "bits": 2,
            "columnwise": True,
        }
        # if args.anaylse_method == "w":
        #     row_layer_range = analysis_result[each]["rowwise"]["w"]["max"] - analysis_result[each]["rowwise"]["w"]["min"]
        #     column_layer_range = analysis_result[each]["columnWise"]["w"]["max"] - analysis_result[each]["columnWise"]["w"]["min"]
        #     std_row = torch.std(row_layer_range)
        #     std_column = torch.std(column_layer_range)
        #     layer_quant_config["columnwise"] = bool(std_column < std_row)
        # elif args.anaylse_method == "wa":
        #     row_layer_range = analysis_result[each]["rowwise"]["wa"]["max"] - analysis_result[each]["rowwise"]["wa"]["min"]
        #     column_layer_range = analysis_result[each]["columnWise"]["wa"]["max"] - analysis_result[each]["columnWise"]["wa"]["min"]
        #     std_row = torch.std(row_layer_range)
        #     std_column = torch.std(column_layer_range)
        #     layer_quant_config["columnwise"] = bool(std_column < std_row)


        if args.anaylse_method == "w":
            layer_range = analysis_result[each]["max"]["w"] - analysis_result[each]["min"]["w"]
            layer_std = analysis_result[each]["std"]["w"]
            

        elif args.anaylse_method == "wa":
            layer_range = analysis_result[each]["max"]["wa"] - analysis_result[each]["min"]["wa"]
            layer_std = analysis_result[each]["std"]["wa"]


        elif args.anaylse_method == "wh":
            layer_range = analysis_result[each]["max"]["wh"] - analysis_result[each]["min"]["wh"]
            layer_std = analysis_result[each]["std"]["wh"]

        elif args.anaylse_method == "wn":
            layer_range = analysis_result[each]["max"]["wn"] - analysis_result[each]["min"]["wn"]
            layer_std = analysis_result[each]["std"]["wn"]

        weight_score.append(layer_range * layer_std)
        model_quant_config[each] = layer_quant_config

    _, weight_score_index = torch.sort(torch.tensor(weight_score))

    mix_bits = [each *  len(weight_score_index) for each in args.mix_bits]
    for i, each in enumerate(weight_score_index):
        model_quant_config[layers[each]]["bits"] = bisect.bisect(mix_bits, i) + 1

    print(model_quant_config)

    # Statistical mixed bits averages
    count = bits_sum = 0
    for each in model_quant_config.keys():
        if "fc" in each:
            count += 4
            bits_sum += 4 * model_quant_config[each]["bits"]
        else:
            count += 1
            bits_sum += model_quant_config[each]["bits"]
    print(f"mix bits avg: {bits_sum/count} bits")

    with open(f"./sensitivity/zeroshot/{model_name}-{args.anaylse_method}.json", "w") as f:
        json.dump(model_quant_config, f)