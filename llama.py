import time
import os
import json

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *


from bcq_quant.quant_model_bcq import quant_model
from lut_gemm.quant import load_lut
from bcq_quant.quantizer import BCQuantizer
from nonLinear_quant import NonLinearQuantizer

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model
    

@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
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
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    if args.layermix:
        import json
        with open("./quant_bit/layerwise.json", "r") as f:
            layer_wbit_dict = json.load(f)
            model_name = str(args.model).split("/")[-1]
            layer_wbit = layer_wbit_dict[model_name]
        print(f"layer_wbit: {layer_wbit}")
    
    if args.linearmix:
        import json
        with open("./quant_bit/linearwise.json", "r") as f:
            linear_wbit = json.load(f)
        print(f"linear_wbit: {linear_wbit}")

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            if args.layermix:
                if args.lut_eval or args.columnwise:
                    gptq[name].quantizer = BCQuantizer(subset[name] if args.lut_eval else nn.Linear(1, 1),
                                                       groupsize=args.groupsize, 
                                                       wbits=layer_wbit[i],
                                                       rounds=args.bcq_round,
                                                       use_bst=args.use_bst, 
                                                       apot_nums=args.apot_nums)
                elif args.non_linear:
                    gptq[name].quantizer = NonLinearQuantizer(subset[name], 
                                                       wbits=layer_wbit[i],
                                                       hyperbits=layer_wbit[i]+2,
                                                       exploreBits=args.exploreBits,
                                                       exploreSplit=args.exploreSplit)
                else:
                    gptq[name].quantizer = Quantizer()
                    gptq[name].quantizer.configure(
                        layer_wbit[i], perchannel=True, sym=args.sym, mse=False, trits=args.trits
                    )
            elif args.linearmix:
                if args.lut_eval or args.columnwise:
                    gptq[name].quantizer = BCQuantizer(subset[name] if args.lut_eval else nn.Linear(1, 1),
                                                       groupsize=args.groupsize, 
                                                       wbits=linear_wbit[name.split(".")[-1]],
                                                       rounds=args.bcq_round,
                                                       use_bst=args.use_bst, 
                                                       apot_nums=args.apot_nums)
                elif args.non_linear:
                    gptq[name].quantizer = NonLinearQuantizer(subset[name], 
                                                       wbits=linear_wbit[name.split(".")[-1]],
                                                       hyperbits=linear_wbit[name.split(".")[-1]]+2,
                                                       exploreBits=args.exploreBits,
                                                       exploreSplit=args.exploreSplit)
                else:
                    gptq[name].quantizer = Quantizer()
                    gptq[name].quantizer.configure(
                        linear_wbit[name.split(".")[-1]], perchannel=True, sym=args.sym, mse=False, trits=args.trits
                    )
            else:
                if args.lut_eval or args.columnwise:
                    gptq[name].quantizer = BCQuantizer(subset[name] if args.lut_eval else nn.Linear(1, 1),
                                                       groupsize=args.groupsize, 
                                                       wbits=args.wbits,
                                                       rounds=args.bcq_round,
                                                       use_bst=args.use_bst, 
                                                       apot_nums=args.apot_nums)
                elif args.non_linear:
                    gptq[name].quantizer = NonLinearQuantizer(subset[name], 
                                                       wbits=args.wbits,
                                                       hyperbits=args.hyperbits,
                                                       exploreBits=args.exploreBits,
                                                       exploreSplit=args.exploreSplit)
                else:
                    gptq[name].quantizer = Quantizer()
                    gptq[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=args.sym, mse=False
                    )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, cache_position = position_ids.squeeze())[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            print('Quantizing ...')
            gptq[name].fasterquant(
                blocksize=128,
                percdamp=args.percdamp, groupsize=args.groupsize, 
                actorder=args.act_order, static_groups=args.static_groups, 
                model_name=str(args.model).split("/")[-1], layer_name=f"{i}.{name}",
                lut_quant=args.lut_eval, non_linear_quant=args.non_linear, columnwise=args.columnwise
            )
            quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
            gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, cache_position = position_ids.squeeze())[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, cache_position = position_ids.squeeze())[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    with open("./quant_bit/llama_ppl.txt", "a") as f:
        f.write(f"model = {str(args.model).split('/')[-1]}, wbits = {args.wbits}, groupsize = {args.groupsize}, lut = {args.lut_eval}, nonLinear = {args.non_linear}, columnwise = {args.columnwise}   :   {ppl.item()}")
        
        if args.non_linear:
            f.write(f"  ||  hyperbits = {args.hyperbits}, exploreBits = {args.exploreBits}, exploreSplit = {args.exploreSplit}")
        if args.lut_eval or args.columnwise:
            f.write(f"  ||  bcq_round = {args.bcq_round}")
            f.write(f"  ||  apot_nums = {args.apot_nums} use_bst = {args.use_bst}")

        if args.layermix:
            import json
            with open("./quant_bit/layerwise.json", "r") as f:
                layer_wbit_dict = json.load(f)
                model_name = str(args.model).split("/")[-1]
                layer_wbit = layer_wbit_dict[model_name]
            f.write(f"  ||  layerMix_wbit = {layer_wbit}")
        
        if args.linearmix:
            import json
            with open("./quant_bit/linearwise.json", "r") as f:
                linear_wbit = json.load(f)
            f.write(f"  ||  linearMix_wbit = {linear_wbit}")
        f.write("\n")

    model.config.use_cache = use_cache

def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
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
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    # bcq quant - LUT-gemm
    parser.add_argument(
        '--bcq', action='store_true', help='Quantize weight with bcq.'
    )
    parser.add_argument(
        '--lut_bench', action='store_true', help='Use Lut-Linear to test latency.'
    )
    parser.add_argument(
        '--lut_eval', action='store_true', help='Use Lut-quantization to evaluate model.'
    )
    parser.add_argument(
        '--bcq_round', type=int, default=5,
        help='Steps to iterate bcq quantization.'
    )
    # non_linear quant - LUT-gemm
    parser.add_argument(
        '--non_linear', action='store_true',
        help='Use non_linear-quantization to evaluate model. Can be converted to LUT type.'
    )
    parser.add_argument(
        '--hyperbits', type=int, default=5,
        help='Use hyperbits linear quant to explore possible non_linear choice.'
    )
    parser.add_argument(
        '--exploreBits', type=int, default=1,
        help='To explore better scale. Start at scale for (hyperbits - exploreBits) and end at scale for (hyperbits + exploreBits).)'
    )
    parser.add_argument(
        '--exploreSplit', type=int, default=20,
        help='To explore better scale. Split the range into (exploreSplit) parts.'
    )

    # columnwise quant
    parser.add_argument(
        '--columnwise', action='store_true',
        help='Use columnwise - bcq - round to power of 2 - quantization to evaluate model. Can be used with new cuda kernel.'
    )
    parser.add_argument(
        '--use_bst', action='store_true',default=False,
        help='Use bst of get BinaryWeight'
    )
    parser.add_argument(
        '--apot_nums', type=int, default=2,
        help='set nums shift weight for quantization.'
    )

    # mix precision
    parser.add_argument(
        '--linearmix', action='store_true',
        help='Whether to use different wbit for different linear type.'
    )
    parser.add_argument(
        '--layermix', action='store_true',
        help='Whether to use different wbit for different layer.'
    )
    args = parser.parse_args()

    model = get_llama(args.model)
    model.eval()
    print(model)

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest and not args.lut_bench:
        tick = time.time()
        if args.bcq:
            print("quantizing with bcq")
            model = quant_model(model, qbits=args.wbits, group_size=args.groupsize)
        else:
            quantizers = llama_sequential(model, dataloader, DEV)
        print("full quantization time: ",time.time() - tick)

    datasets = ['wikitext2', 'ptb'] 
    if args.new_eval:
        datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama_eval(model, testloader, DEV)

    if args.save:
        llama_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)
