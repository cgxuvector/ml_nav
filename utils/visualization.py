import matplotlib.pyplot as plt
import numpy as np
from utils import mapper


def rolling_average(data, window_size):
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(np.ones_like(data), kernel)
    return smooth_data[: -window_size + 1]


def plot_line_chart(data, name, x_label, y_label, smooth_win_size, color, start, end):
    x = np.arange(data.shape[0])
    data_smooth = rolling_average(data, smooth_win_size)
    plt.title(name)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.plot(x[start:end+1], data[start:end+1], color[0], linewidth=4)
    plt.plot(x[start:end+1], data_smooth[start:end+1], color[1], linewidth=4)
    plt.show()


if __name__ == '__main__':
    root_dir = '../results/4-3/'
    data_name = 'goal_conditioned_double_dqn_5x5_ep_200_return.npy'
    d = np.load(root_dir + data_name)
    start = 0
    end = d.shape[0]
    # plot_line_chart(d, f"Sub-goal {end}", "Episode", "Distance", 50, ['lightsalmon', '-r'], start, end)

    plot_line_chart(d, "Double DQN in 5 x 5 Maze ", "Episode", "Discounted cumulative rewards", 100, ['lightgreen', '-g'], start, end)