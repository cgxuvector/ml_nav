import matplotlib.pyplot as plt
import numpy as np
from utils import mapper


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
    plt.plot(x, data, color[0], linewidth=2)
    plt.plot(x, data_smooth, color[1], linewidth=2)
    plt.show()


def success_rate():
    maze_size = [5, 7, 9, 11, 13]
    value = [6, 10, 14, 18, 22]
    for size, val in zip(maze_size, value):
        # dist_data = np.load(f'../results/4-27/double_dqn_{size}x{size}_ep_1000_b64_14_distance.npy')
        # len_data = np.load(f'../results/4-27/double_dqn_{size}x{size}_ep_1000_b64_14_length.npy')
        goal_dist_data = np.load(f'../results/4-30/goal_ddqn_{size}x{size}_true_state_distance.npy')
        goal_len_data = np.load(f'../results/4-30/goal_ddqn_{size}x{size}_true_state_length.npy')
        dist_data = np.load(f'../results/4-30/ddqn_{size}x{size}_true_state_distance.npy')
        len_data = np.load(f'../results/4-30/ddqn_{size}x{size}_true_state_length.npy')
        success_count = 0
        total_count = dist_data.shape[0]
        last_count = 0
        count = 0
        idx = 0
        success_rate_list = []
        while count < total_count:
            if dist_data[count] <= 10 and len_data[count] < val:
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

        total_count = goal_dist_data.shape[0]
        last_count = 0
        count = 0
        idx = 0
        success_rate_list = []
        while count < total_count:
            if goal_dist_data[count] <= 10 and goal_len_data[count] < val:
                success_count += 1
            if (count + 1) % 100 == 0:
                success_rate_list.append(success_count / (count - last_count))
                success_count = 0
                last_count = count
                idx += 1
            count += 1
        plt.plot(range(len(success_rate_list)), success_rate_list, 'r-', linewidth=2)
        plt.show()


def plot_compare(data_true_state, data_obs_decal_1, name, x_label, y_label, smooth_win_size, color):
    # load the x label for each data
    x_true_state = np.arange(data_true_state.shape[0])
    x_obs_1 = np.arange(data_obs_decal_1.shape[0])
    # x_obs_10 = np.arange(data_obs_decal_10.shape[0])
    # smooth the data
    true_state_data_smooth = rolling_average(data_true_state, smooth_win_size)
    obs_1_data_smooth = rolling_average(data_obs_decal_1, smooth_win_size)
    # obs_10_data_smooth = rolling_average(data_obs_decal_10, smooth_win_size)

    fig, arr = plt.subplots()
    plt.title(name)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    # plt.plot(x_true_state, data_true_state, color[0], linewidth=2)
    line1, = arr.plot(x_true_state, true_state_data_smooth, color[1], linewidth=2)
    # plt.plot(x_obs_1, data_obs_1, color[2], linewidth=2)
    line2, = plt.plot(x_obs_1, obs_1_data_smooth, color[3], linewidth=2)
    plt.legend([line1, line2], ['true state', 'panorama obs'])
    # plt.plot(x_obs_10, data_obs_10, color[4], linewidth=2)
    # plt.plot(x_obs_10, obs_10_data_smooth, color[5], linewidth=2)
    plt.show()

    plot_line_chart(data_true_state, name + ': true state', x_label, y_label, smooth_win_size, ['lightgreen', 'green'])
    plot_line_chart(data_obs_decal_1, name + ': panorama obs', x_label, y_label, smooth_win_size, ['lightsalmon', 'red'])


if __name__ == '__main__':
    root_dir = '../results/5-3/'
    data_name = 'test_goal_return.npy'
    # # # policy_name = 'ddqn_5x5_true_state_double_policy_return.npy'
    d = np.load(root_dir + data_name)
    # # # pd = np.load(root_dir + policy_name)
    #
    plot_line_chart(d, "goal conditioned 5x5", "Episode", "Discounted Return", 100, ['lightgreen', '-g'])
    # # success_rate()
    # plt.title('Evaluate the learned policy every 100 episodes')
    # plt.xlabel('every 100 episode')
    # plt.ylabel('return')
    # plt.plot(range(pd.shape[0]), pd, 'r-', linewidth=2)
    # plt.show()

    # size = 13
    # data_true_state = np.load(root_dir + f'ddqn_{size}x{size}_true_state_double_long_return.npy')
    # data_obs_1 = np.load(root_dir + f'ddqn_{size}x{size}_obs_decal_1_m20000_double_return.npy')
    # # data_obs_10 = np.load(root_dir + f'ddqn_{size}x{size}_obs_decal_10_double_return.npy')
    # plot_compare(data_true_state, data_obs_1, f'Learning curve of maze {size}x{size}', 'Episode', 'Discounted Return', 100, ['lightgreen', 'g', 'lightsalmon', 'red', 'lightblue', 'blue'])