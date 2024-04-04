import os
import json

import torch
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

data_dir = "/data/hyou37/yipin/program/gptq/sensitivity/"
def get_data(file_name):
    quant_loss = {}
    with open(data_dir + file_name, 'r') as f:
        for line in f:
            layer, loss = line.split(': ')
            quant_loss[layer] = float(loss)
    return quant_loss

model_name = "opt-1.3b"
sensitivity = get_data(f"{model_name}-3bits-8column.txt")
layers_name = list(sensitivity.keys())
error_reduce_perbit = 0.4
max_avg_bit = 2.2
def objective(config):
    quant_error = 0
    bits = 0
    num = 0
    for layer, loss in sensitivity.items():
        quant_error += loss * (error_reduce_perbit ** (config[layer] - 3))
        num += 4 if "fc" in layer else 1
        bits += 4*config[layer] if "fc" in layer else config[layer]
    
    if bits / num > max_avg_bit:
        return {"error": 1000000000000}

    
    return {"error": quant_error}
    

search_space = {}
for layer in layers_name:
    search_space[layer] = tune.choice([1,2,3,4])

tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="error",
        mode="min",
        num_samples=1000,
        scheduler=ASHAScheduler(),
    ),
    param_space=search_space,
    run_config=train.RunConfig(
        name="opt-quant-error-search",
        # resources_per_trial={"cpu": 1, "gpu": 0},
        verbose=0,
    ),
)
results = tuner.fit()
best_result = results.get_best_result().config
print("Best config is:", best_result)

out_dict = {}
for each in best_result:
    out_dict["model.decoder.layers." + each] = {"bits": best_result[each], "columnwise": True}
with open(os.path.join(data_dir, f"zeroshot/{model_name}-search.json"), 'w') as f:
    json.dump(out_dict, f)