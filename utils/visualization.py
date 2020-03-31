import matplotlib.pyplot as plt
import numpy as np


def rolling_average(data, window_size):
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(np.ones_like(data), kernel)
    return smooth_data[: -window_size + 1]


def plot_line_chart(data, name, x_label, y_label, smooth_win_size, color, step):
    x = np.arange(data.shape[0])
    data_smooth = rolling_average(data, smooth_win_size)
    plt.title(name)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.plot(x[0:step+1], data[0:step+1], color[0], linewidth=4)
    plt.plot(x[0:step+1], data_smooth[0:step+1], color[1], linewidth=4)
    plt.show()


if __name__ == '__main__':
    root_dir = '../results/3-30/'
    data_name = 'conditioned_double_dqn_fixed_goal_7_seed_1_return.npy'
    d = np.load(root_dir + data_name)
    # plot_line_chart(d, "Non Goal Conditioned Double DQN Seed 1 ", "Episode", "Distance", 100, ['lightsalmon', '-r'], d.shape[0])
    plot_line_chart(d, "Non Goal Conditioned Double DQN Seed 1 ", "Episode", "Distance", 100, ['lightsalmon', '-r'], 200)