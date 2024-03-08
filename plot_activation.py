import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_distribution(value, file_path = "./plot.png", name_x = "tokens", name_y = "channels"):
    # assert len(value.shape) == 3, f"Expected 3D tensor, got {value.shape} instead."
    # B, N, M = value.shape
    # value = value.abs().mean(dim=0)[100:200,:].flatten().detach().cpu().numpy() + 1e-6
    # N = 100
    print("plot_distribution to", file_path)
    assert len(value.shape) == 2, f"Expected 3D tensor, got {value.shape} instead."
    N, M = value.shape
    value = value.abs().flatten().detach().cpu().numpy()

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    _x = np.arange(N)
    _y = np.arange(M)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    
    colors = plt.cm.viridis(value / np.max(value))


    # 绘制柱状图
    ax.bar3d(x, y, np.zeros(len(value)), 1, 1, value, color=colors)

    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    ax.set_zlabel('mean(abs(x))')
    # ax.set_zscale('log')

    # On the y-axis let's only label the discrete values that we have data for.
    plt.savefig(file_path)
    plt.close()
    print("plot_distribution done")



def plot_distribution2d(value, file_path = "./plot.png"):

    print("plot_distribution to", file_path)
    assert len(value.shape) == 2, f"Expected 3D tensor, got {value.shape} instead."
    N, M = value.shape

    # row wise
    plt.subplot(1, 2, 1)
    x = np.arange(N)
    max_label = torch.max(value).cpu().numpy()
    min_label = torch.min(value).cpu().numpy()

    max_value = torch.max(value, dim=1)[0].cpu().numpy()
    min_value = torch.min(value, dim=1)[0].cpu().numpy()
    heights = max_value-min_value

    plt.bar(x, heights, bottom=min_value)
    plt.ylim(min_label, max_label)
    plt.title('row wise')

    # column wise
    plt.subplot(1, 2, 2)
    x = np.arange(M)
    max_value = torch.max(value, dim=0)[0].cpu().numpy()
    min_value = torch.min(value, dim=0)[0].cpu().numpy()
    heights = max_value-min_value
    plt.bar(x, heights, bottom=min_value)
    plt.title('column wise')
    plt.ylim(min_label, max_label)

    plt.tight_layout()

    plt.savefig(file_path)
    plt.close()