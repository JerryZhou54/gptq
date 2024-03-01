import time
import math

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *

from bcq_quant.quant_model_bcq import quant_model
from lut_gemm.quant import load_lut
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

        subset = find_layers(layer)
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
                        args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
                    )

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
            print('Quantizing ...')
            gptq[name].fasterquant(
                blocksize=128,
                percdamp=args.percdamp, groupsize=args.groupsize, 
                actorder=args.act_order, static_groups=args.static_groups, 
                model_name=str(args.model).split("/")[-1], layer_name=f"{i}.{name}",
                lut_quant=args.lut_eval, non_linear_quant=args.non_linear, columnwise=args.columnwise
            )
            quantizers['model.decoder.layers.%d.%s' % (i, name)] = gptq[name].quantizer
            gptq[name].free()
        
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

@torch.no_grad()
def opt_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

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
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        # print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        print("draw distribution layer ", i)
        plot_distribution(outs, file_path=f"./plot_activation/layer_{i}.png")
        print("draw distribution done, layer ", i)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
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
    with open("./quant_bit/ppl.txt", "a") as f:
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

# TODO: perform packing on GPU
def opt_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers, faster=args.faster_kernel)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

def load_quant3(model, checkpoint):
    import transformers
    from transformers import OPTConfig, OPTForCausalLM 
    config = OPTConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = OPTForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['model.decoder.project_out', 'model.decoder.project_in', 'lm_head']:
        if name in layers:
            del layers[name]
    make_quant3(model, layers, faster=args.faster_kernel)

    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = model.config.max_position_embeddings
    print('Done.')

    return model

def opt_multigpu(model, gpus):
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(gpus[0])
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(gpus[0])
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(gpus[0])
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(gpus[-1])
    if hasattr(model.model.decoder, 'final_layer_norm') and model.model.decoder.final_layer_norm:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(gpus[-1])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {'mask': None}

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.decoder.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus

@torch.no_grad()
def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV) # [1,128]
    torch.cuda.synchronize()

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
            

    attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
    times = []

    for _ in range(100): #warmup
        out = model(
            input_ids[:, 0].reshape((1,-1)),
            past_key_values=cache['past'],
            attention_mask=attention_mask[:, :1].reshape((1, -1))
        )
        sync()

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False, with_stack=True) as prof:
        out = model(
            input_ids[:, 0].reshape((1,-1)),
            past_key_values=cache['past'],
            attention_mask=attention_mask[:, :1].reshape((1, -1))
        )
        sync()
    prof.export_chrome_trace("./weight/profile/torch_trace.json")


    for i in range(input_ids.numel()):
        tick = time.time()
        out = model(
            input_ids[:, i].reshape((1,-1)),
            past_key_values=cache['past'],
            attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
        )
        sync()
        times.append(time.time() - tick)
        # print(i, times[-1])
        if check and i != input_ids.numel() - 1:
            tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
        cache['past'] = list(out.past_key_values)
        del out
    sync()
    import numpy as np
    print('Median:', np.median(times))
    if check:
        print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())


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
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[1, 2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
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
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to use the new faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--static-groups', action='store_true', default=True,
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

    if args.load:
        model = load_quant3(args.model, args.load)
    elif args.lut_bench:
        model = load_lut(args.model, args.load, wbit=args.wbits, group_size=args.groupsize)
    else:
        model = get_opt(args.model)
        model.eval()
    print(model)

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest and not args.load and not args.lut_bench:
        tick = time.time()
        if args.bcq:
            model = quant_model(model, qbits=args.wbits, group_size=args.groupsize)
        else:
            quantizers = opt_sequential(model, dataloader, DEV)
        print("full quantization time: ",time.time() - tick)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            opt_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        
        input_ids = next(iter(dataloader))[0][:, :args.benchmark]
        print(input_ids.shape)
        benchmark(model, input_ids, check=args.check)
            
    if args.load or args.lut_bench:
        exit()

    datasets = ['wikitext2', 'ptb'] 
    if args.new_eval:
      datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets: 
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV)
    # print(args.dataset)
    # opt_eval(model, testloader, DEV)

    if args.save:
        opt_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save) 
