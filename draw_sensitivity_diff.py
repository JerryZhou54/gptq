import numpy as np
import matplotlib.pyplot as plt

data_dir = "./sensitivity/"
baseline_sensitive_file = "opt-125m-3bit.txt"
compared_sensitive_file = "opt-125m-4bit.txt"


def get_data(file_name):
    quant_loss = {}
    with open(data_dir + file_name, 'r') as f:
        for line in f:
            layer, loss = line.split(': ')
            quant_loss[layer] = float(loss)
    return quant_loss

def main():
    baseline_data = get_data(baseline_sensitive_file)
    compared_data = get_data(compared_sensitive_file)
    assert len(baseline_data) == len(compared_data), "should compare the sensitivity of the same model"
    assert list(baseline_data.keys()) == list(compared_data.keys()), "should compare the sensitivity of the same model"
    
    layers_name = list(baseline_data.keys())
    diff_data = np.array(list(compared_data.values())) - np.array(list(baseline_data.values()))

    x_pos = np.arange(len(layers_name))

    # draw

    plt.figure(figsize=(20, 5))
    plt.bar(x_pos, diff_data, align='center', alpha=0.5)
    plt.xticks(x_pos, layers_name, rotation=270)
    plt.ylabel('sensitivity difference')
    plt.title(f'the difference of sensitivity between {baseline_sensitive_file}(base) and {compared_sensitive_file}(target)')
    plt.savefig(f'./sensitivity/diff.png')
    plt.close()

if __name__ == "__main__":
    main()