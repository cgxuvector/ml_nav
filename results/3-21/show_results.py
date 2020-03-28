import matplotlib.pyplot as plt
import numpy as np


def rolling_average(data, window_size):
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(np.ones_like(data), kernel)
    return smooth_data[: -window_size + 1]


if __name__ == '__main__':
    returns_1 = np.load("dqn_random_init_goal_return.npy")
    returns_smooth_1 = rolling_average(returns_1, 100)
    plt.plot(np.arange(returns_1.shape[0])[0:2000], returns_1[0:2000], 'lightsalmon', linewidth=4)
    plt.plot(np.arange(returns_1.shape[0])[0:2000], returns_smooth_1[0:2000], '-r', linewidth=4)
    plt.title("Double DQN with 5x5 maze and fixed start and goal positions")
    plt.xlabel("Episode")
    plt.ylabel("Distance")
    plt.show()