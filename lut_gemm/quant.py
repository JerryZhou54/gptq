import numpy as np
import torch
import torch.nn as nn

import lutgemm

class LutLinear(nn.Module): 
    def __init__(self, infeatures:int,
                        outfeatures:int,
                        group_size:int = 128,
                        wbit:int = 3,):

        super().__init__()

        self.M = 1
        self.K = infeatures
        self.N = outfeatures
        self.group_size = group_size
        self.wbit = wbit
        self.num_groups = self.K // group_size

        self.register_buffer('binaryWeight', torch.zeros((self.K //32 , wbit, self.N), dtype=torch.int32))
        self.register_buffer('alpha', torch.zeros((self.num_groups, wbit, self.N), dtype=torch.half))
        self.register_buffer('q_bias', torch.zeros((self.num_groups, self.N), dtype=torch.half))

        self.bWeight_cal = None
        self.alpha_cal = None
        self.q_bias_cal = None

        # self.bias = None
        self.register_buffer('bias', torch.zeros(self.N, dtype=torch.half))

    def pack(self, linear):
        # TODO: do quantization here
        self.bias = linear.bias.clone() if linear.bias is not None else torch.zeros(self.N, dtype=torch.half)
        pass

    def parsing(self):
        # device = self.binaryWeight.device
        # assert device != torch.device("cpu"), "Device should be cuda"
        # device = int(str(device).split(":")[-1])
        # self.bWeight_cal, self.alpha_cal, self.q_bias_cal = lutgemm.parsing(self.binaryWeight, self.alpha.view(-1), self.K, self.N, self.wbit, False, self.num_groups, self.q_bias.view(-1), device)

        self.bWeight_cal, self.alpha_cal, self.q_bias_cal = self.binaryWeight.view(-1), self.alpha.view(-1), self.q_bias.view(-1)

    def forward(self, x):
        if self.bWeight_cal is None or self.alpha_cal is None or self.q_bias_cal is None:
            self.parsing()

        assert x.shape[-1] == x.numel(), "Only support batch_size = 1 and 2 dimension input"

        outshape = list(x.shape)
        output = self.bias.clone()
        outshape[-1] = self.bias.numel()
        dtype = x.dtype

        lutgemm.lutgemm_compute(output, self.bWeight_cal, self.alpha_cal, self.q_bias_cal, x, self.N, self.K, self.wbit, self.num_groups)

        output = output.to(dtype).reshape(outshape)
        return output

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}, wbit={}, group_size={}'.format(
            self.K, self.N, self.bias is not None, self.wbit, self.group_size
        )

def make_lut(module, names, name='', wbit = 3, group_size = 128):
    if isinstance(module, LutLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(module, attr, LutLinear(tmp.in_features, tmp.out_features, group_size, wbit))
    for name1, child in module.named_children():
        make_lut(child, names, name + '.' + name1 if name != '' else name1, wbit, group_size)

def load_lut(model, checkpoint='', wbit = 3, group_size = 128):

    def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res

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
    make_lut(model, layers, wbit = wbit, group_size = group_size)

    if checkpoint != '':
        print('Loading model ...')
        model.load_state_dict(torch.load(checkpoint))
        
    model.seqlen = model.config.max_position_embeddings
    print('Done.')

    return model


