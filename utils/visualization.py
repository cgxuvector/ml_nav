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


def load_data_from_runs(data_dir, size, use_random=False, use_goal=False, use_obs=False):
    # list to store all the data
    data_load_list = []
    # maximal data length
    max_len = 0
    # load the data for all the runs
    for r in range(run_num):
        # get the name
        if not use_goal:
            if not use_obs:
                file_name = f'ddqn_{size}x{size}_true_state_her_double_seed_{r}_return.npy'
            else:
                file_name = f'ddqn_{size}x{size}_panorama_obs_double_seed_{r}_return.npy'
        else:
            if use_random:
                if not use_obs:
                    file_name = f'random_goal_ddqn_{size}x{size}_true_state_her_double_seed_{r}_return.npy'
                else:
                    file_name = f'random_goal_ddqn_{size}x{size}_obs_double_her_seed_{r}_return.npy'
            else:
                if not use_obs:
                    file_name = f'goal_ddqn_{size}x{size}_true_state_her_double_seed_{r}_return.npy'
                else:
                    file_name = f'goal_ddqn_{size}x{size}_panorama_obs_double_seed_{r}_return.npy'
        print(file_name)
        # load the data
        data = np.load(data_dir + file_name)
        # save the data
        data_load_list.append(data)
        # update the maximal length
        if data.shape[0] > max_len:
            max_len = data.shape[0]
    return data_load_list, max_len


# compute the mean and standard error of the returns af all the runs
def compute_mean_and_std_error(data, max_len):
    # combine the list of return in to a combined list
    combined_data_list = []
    # scan each episode
    for i in range(max_len):
        elem_list = []
        # scan each run
        for j in range(run_num):
            if i < data[j].shape[0]:
                elem_list.append(data[j][i])
        combined_data_list.append(np.array(elem_list))
    # compute the mean
    mean_val = [np.mean(elem) for elem in combined_data_list]
    # compute the standard error of each element
    std_err_val = [np.std(elem) / len(elem) for elem in combined_data_list]
    return mean_val, std_err_val


# plot the results
def plot_mean_std_error(name, x_label, y_label, m_list, std_list, opt_list, w_size):
    # rolling average
    optimal_val = np.array(opt_list)
    mean_val = rolling_average(np.array(m_list), w_size)
    std_val = rolling_average(np.array(std_list), win_size)

    # set the settings
    t = range(len(m_list))
    mu = mean_val
    sigma_err = std_val

    # plot the learning curves
    fig, ax = plt.subplots(1)
    ax.set_title(name)
    ax.set_ylim(mu.min(), 0)
    ax.plot(t, optimal_val, label='oracle', color='black', ls='--')
    ax.plot(t, mu, lw=1, label='mean', color='green')
    ax.fill_between(t, mu + sigma_err, mu - sigma_err, lw=2, facecolor='green', alpha=0.5)
    ax.legend(loc='lower right')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()
    plt.show()


# compare true state and observation
def plot_compared_mean_std_error(name, x_label, y_label,
                                 m_1_list, std_1_list, opt_list,
                                 m_2_list, std_2_list,
                                 w_size):
    # rolling average
    optimal_val = np.array(opt_list)
    mean_1_val = rolling_average(np.array(m_1_list), w_size)
    mean_2_val = rolling_average(np.array(m_2_list), win_size)
    std_1_val = rolling_average(np.array(std_1_list), win_size)
    std_2_val = rolling_average(np.array(std_2_list), win_size)

    # set the settings
    t1 = range(len(m_1_list))
    t2 = range(len(m_2_list))
    mu1 = mean_1_val
    mu2 = mean_2_val
    sigma1_err = std_1_val
    sigma2_err = std_2_val

    # plot the learning curves
    fig, ax = plt.subplots(1)
    ax.set_title(name)
    ax.set_ylim(min(mu1.min(), mu2.min()), 0)
    ax.plot(range(len(optimal_list)), optimal_val, label='oracle', color='black', ls='--')
    ax.plot(t1, mu1, lw=1, label='True state', color='red')
    ax.fill_between(t1, mu1 + sigma1_err, mu1 - sigma1_err, lw=2, facecolor='red', alpha=0.5)
    ax.plot(t2, mu2, lw=1, label='Panorama obs', color='green')
    ax.fill_between(t2, mu2 + sigma2_err, mu2 - sigma2_err, lw=2, facecolor='green', alpha=0.5)
    ax.legend(loc='lower right')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()
    plt.show()


# compute the oracle return (optimal value)
def compute_oracle(size, local_policy=False):
    map = mapper.RoughMap(size, 0, 3)
    if not local_policy:
        optimal_step = len(map.path) - 1
    else:
        optimal_step = 2
    gamma = 0.99
    rewards = [-1] * (optimal_step - 1)
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
    return G


if __name__ == '__main__':
    # experiment settings
    root_dir = '../results/5-12/'
    maze_size = 5
    plot_name = f'Random Goal-conditioned Double DQN with HER in maze {maze_size}x{maze_size}'
    # optimal value
    oracle_val = compute_oracle(maze_size, local_policy=True)
    run_num = 1
    win_size = 50
    # load data
    # data_1_list, max_ep_1_len = load_data_from_runs(root_dir, maze_size, use_random=True, use_goal=True, use_obs=False)
    data_2_list, max_ep_2_len = load_data_from_runs(root_dir, maze_size, use_random=True, use_goal=True, use_obs=True)
    # compute the mean and std error
    # mean_1_list, std_err_1_list = compute_mean_and_std_error(data_1_list, max_ep_1_len)
    mean_2_list, std_err_2_list = compute_mean_and_std_error(data_2_list, max_ep_2_len)
    # plot the results
    # optimal_list = [oracle_val] * max(max_ep_1_len, max_ep_2_len)
    optimal_list = [oracle_val] * max_ep_2_len
    # plot_compared_mean_std_error(plot_name,
    #                              'Episode', r'Discounted return $S_{init}$',
    #                              mean_1_list, std_err_1_list, optimal_list,
    #                              mean_2_list, std_err_2_list,
    #                              win_size)
    plot_mean_std_error(plot_name,
                        'Episode', r'Discounted return $S_{init}$',
                        mean_2_list, std_err_2_list, optimal_list,
                        win_size)

    # data_name = 'test_goal_return.npy'
    # policy_name = 'ddqn_5x5_true_state_double_policy_return.npy'
    # d = np.load(root_dir + data_name)
    # pd = np.load(root_dir + policy_name)

    # plot_line_chart(d, "goal conditioned 5x5", "Episode", "Discounted Return", 100, ['lightgreen', '-g'])
    # success_rate()
    # plt.title('Evaluate the learned policy every 100 episodes')
    # plt.xlabel('every 100 episode')
    # plt.ylabel('return')
    # plt.plot(range(pd.shape[0]), pd, 'r-', linewidth=2)
    # plt.show()

    # size = 13
    # data_true_state = np.load(root_dir + f'ddqn_{size}x{size}_true_state_double_long_return.npy')
    # data_obs_1 = np.load(root_dir + f'ddqn_{size}x{size}_obs_decal_1_m20000_double_return.npy')
    # data_obs_10 = np.load(root_dir + f'ddqn_{size}x{size}_obs_decal_10_double_return.npy')
    # plot_compare(data_true_state, data_obs_1, f'Learning curve of maze {size}x{size}', 'Episode', 'Discounted Return', 100, ['lightgreen', 'g', 'lightsalmon', 'red', 'lightblue', 'blue'])