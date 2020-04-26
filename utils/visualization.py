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


def success_rate():
    maze_size = [11]
    for size in maze_size:
        dist_data = np.load(f'../results/4-23/double_dqn_{size}x{size}_ep_1000_b64_m80000_distance.npy')
        len_data = np.load(f'../results/4-23/double_dqn_{size}x{size}_ep_1000_b64_m80000_length.npy')
        success_count = 0
        total_count = dist_data.shape[0]
        last_count = 0
        count = 0
        idx = 0
        success_rate_list = []
        while count < total_count:
            if dist_data[count] <= 10 and len_data[count] < 1000:
                success_count += 1
            if (count+1) % 100 == 0:
                success_rate_list.append(success_count / (count - last_count))
                success_count = 0
                last_count = count
                idx += 1
            count += 1
        plt.title(f"Navigation success rate of maze {size} x {size} every 100 epochs")
        plt.plot(range(len(success_rate_list)), success_rate_list, 'b-', linewidth=2)
        plt.xlabel("every 100 epochs ")
        plt.ylabel("success rate (%)")
        plt.show()


if __name__ == '__main__':
    root_dir = '../results/4-23/'
    data_name = 'double_dqn_11x11_ep_1000_b64_m80000_return.npy'
    d = np.load(root_dir + data_name)
    start = 0
    end = d.shape[0]
    # plot_line_chart(d, f"Sub-goal {end}", "Episode", "Distance", 50, ['lightsalmon', '-r'], start, end)

    plot_line_chart(d, "Double DQN in 11 x 11 Maze with true state", "Episode", "Discounted Return", 100, ['lightgreen', '-g'], start, end)
    success_rate()