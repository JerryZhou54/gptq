import matplotlib.pyplot as plt

# import data
model_name = "opt-6.7b"
model_name_3bit = f"{model_name}-3bit"
model_name_4bit = f"{model_name}-4bit"

def to_layerwise(quant_loss):
    layerwise = {}
    for layer, loss in quant_loss.items():
        layer = layer.split('.')[0]
        if layer not in layerwise:
            layerwise[layer] = loss
        else:
            layerwise[layer] += loss
    return layerwise

def to_linearwise(quant_loss):
    linearwise = {}
    for layer, loss in quant_loss.items():
        layer = layer.split('.')[-1]
        if layer not in linearwise:
            linearwise[layer] = loss
        else:
            linearwise[layer] += loss
    return linearwise

def get_data(model_name):
    quant_loss = {}
    with open(f'./sensitivity/{model_name}.txt', 'r') as f:
        for line in f:
            layer, loss = line.split(': ')
            quant_loss[layer] = float(loss)
    return quant_loss

loss_3bit = get_data(model_name_3bit)
loss_4bit = get_data(model_name_4bit)


layerwise_3bit = to_layerwise(loss_3bit)
linearwise_3bit = to_linearwise(loss_3bit)
layerwise_4bit = to_layerwise(loss_4bit)
linearwise_4bit = to_linearwise(loss_4bit)

# plot bar chart
plt.figure(figsize=(10, 5))
plt.bar(list(layerwise_3bit.keys()), list(layerwise_3bit.values()), align='center', label='3-bit')
plt.bar(list(layerwise_4bit.keys()), list(layerwise_4bit.values()), align='center', label='4-bit')
plt.xlabel('Layer')
plt.ylabel('quantization loss')
plt.legend(['3-bit', '4-bit'])
plt.title(f'{model_name}-Layer-wise quantization loss')
plt.savefig(f'./sensitivity/{model_name}-layerwise.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(list(linearwise_3bit.keys()), list(linearwise_3bit.values()), align='center', label='3-bit')
plt.bar(list(linearwise_4bit.keys()), list(linearwise_4bit.values()), align='center', label='4-bit')
plt.xlabel('Layer')
plt.ylabel('quantization loss')
plt.legend(['3-bit', '4-bit'])
plt.title(f'{model_name}-LinearType-wise quantization loss')
plt.savefig(f'./sensitivity/{model_name}-linearwise.png')
plt.close()