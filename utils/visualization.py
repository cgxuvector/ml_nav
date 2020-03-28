import matplotlib.pyplot as plt
import numpy as np


def rolling_average(data, window_size):
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(np.ones_like(data), kernel)
    return smooth_data[: -window_size + 1]


def plot_line_chart(data, name, x_label, y_label, smooth_win_size, color):
    x = np.arange(data.shape[0])
    data_smooth = rolling_average(data, smooth_win_size)
    plt.title(name)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.plot(x, data, color[0], linewidth=4)
    plt.plot(x, data_smooth, color[1], linewidth=4)
    plt.show()


if __name__ == '__main__':
    root_dir = '../results/3-22/'
    data_name = 'dqn_random_init_goal_distance.npy'
    d = np.load(root_dir + data_name)
    plot_line_chart(d, "Vanilla DQN: 11x11 maze with fixed start and random goal positions", "episode", "Distance", 100, ['lightsalmon', '-r'])


