import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_distribution(value, file_path = "./plot.png", name_x = "tokens", name_y = "channels"):
    assert len(value.shape) == 3, f"Expected 3D tensor, got {value.shape} instead."
    B, N, M = value.shape
    value = value.abs().mean(dim=0)[:100,:].flatten().detach().cpu().numpy() + 1e-6
    N = 100

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    _x = np.arange(N)
    _y = np.arange(M)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    
    colors = plt.cm.viridis(value / max(value))


    # 绘制柱状图
    ax.bar3d(x, y, np.zeros(len(value)), 1, 1, value, color=colors)

    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    ax.set_zlabel('mean(abs(x))')
    # ax.set_zscale('log')

    # On the y-axis let's only label the discrete values that we have data for.
    plt.savefig(file_path)
