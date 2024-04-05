import os
import yaml
from nni.experiment import Experiment


CLEAR = False
port = 8080
model_name = "opt-125m"
experiment_id = f"{model_name}-mixbit-e2e"
CUDA_DEVICE = 0

def get_data(model_name):
    quant_loss = {}
    with open(f'./sensitivity/{model_name}-3bits-8column.txt', 'r') as f:
        for line in f:
            layer, loss = line.split(': ')
            quant_loss[layer] = float(loss)
    return quant_loss

layer_names = list(get_data(model_name).keys())
search_space = {}
for layer in layer_names:
    search_space[layer] = {}
    search_space[layer]["_type"] = "choice"
    search_space[layer]["_value"] = [1,2,3,4]
    # search_space[layer + "/columnwise"] = {'_type': 'choice', '_value': [True, False]},

command = f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE} python opt.py \
    facebook/{model_name} \
    wikitext2 \
    --wbits 2 \
    --groupsize -1 \
    --columnwise \
    --bcq_round 50 \
    --apot_nums 3 \
    --block_quant \
    --use_bst \
    --mixbit 2.2 \
    --nni"

experiment_dir = "./experiments/" + experiment_id
if CLEAR:
    os.system("rm -r " + experiment_dir+"/*")
experiment = Experiment('local')
experiment.config.experiment_name = f'mixbit quant search on {model_name}'
experiment.id = experiment_id

# experiment.resume(experiment_id=experiment_id, port=port)
experiment.config.trial_command = command
experiment.config.experiment_working_directory = experiment_dir
experiment.config.trial_code_directory = './'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Evolution'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.tuner.class_args['population_size'] = 100
experiment.config.max_trial_number = 1000
experiment.config.trial_concurrency = 3

experiment.run(port)

input()
experiment.stop()