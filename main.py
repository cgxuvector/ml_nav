"""
    Script to train a RL agent
"""
import argparse
from model.DQN import DQN
from model.TD3 import TD3
from collections import namedtuple
import gym
import numpy as np
import matplotlib.pyplot as plt
gym.logger.set_level(40)


def dqn_parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn_mode", type=str, default="vanilla", help="Mode of the DQN: vanilla or double")
    parser.add_argument("--clip_grad", type=bool, default=False, help="whether enable the grad clip")
    parser.add_argument("--save_name", type=str)
    return parser.parse_args()


def td3_parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str)
    return parser.parse_args()


def run_DQN():
    # input params
    input_args = dqn_parse_input()

    # load the CarPole environment
    env = gym.make('CartPole-v1')
    # obtain the params for training
    state_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    hidden_dim = 512
    transition_config = namedtuple("transition", ["state", "action", "next_state", "reward", "done"])
    batch_size = 64
    replay_buffer_size = 200_000

    # create a DQN
    myDQN = DQN(env,
                state_dim,
                action_num,
                hidden_dim,
                replay_buffer_size,
                batch_size,
                transition_config=transition_config,
                max_time_steps=51_000,
                learning_rate=1e-3,
                start_update_step=1000,
                target_update_frequency=8,
                policy_update_frequency=4,
                gamma=0.99,
                eps_start=1.0,
                dqn_mode=input_args.dqn_mode,
                gradient_clip=input_args.clip_grad,
                save_name=input_args.save_name,
                soft_target_update=True
                )

    # train the DQN
    myDQN.train()
    # myDQN.eval("offline", "results/model/dqn_vanilla_clip_new.pt")

    dqn_return = np.load("results/return/dqn_vanilla_return.npy")
    smooth_data = DQN.rolling_average(dqn_return, 25)

    x = np.arange(dqn_return.shape[0])
    plt.plot(x, dqn_return, 'salmon')
    plt.plot(x, smooth_data, 'r')
    plt.show()


def run_TD3():
    # input params
    input_args = td3_parse_input()

    # load the CarPole environment
    env = gym.make('CartPole-v1')
    # obtain the params for training
    state_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    hidden_dim = 512
    transition_config = namedtuple("transition", ["state", "action", "next_state", "reward", "done"])
    batch_size = 64
    replay_buffer_size = 200_000

    # create a TD3
    myTD3 = TD3(env,
                state_dim,
                hidden_dim,
                action_num,
                start_time_steps=10_000,
                max_time_steps=510_000,
                buffer_size=replay_buffer_size,
                population_num=0,
                batch_size=batch_size,
                actor_update_frequency=2,
                critic_update_frequency=1,
                save_name=input_args.save_name,
                tau=0.005
                )

    # train the TD3
    # myTD3.train()
    td3_return = np.load("results/return/td3_test_return.npy")
    smooth_data = DQN.rolling_average(td3_return, 25)

    x = np.arange(td3_return.shape[0])
    plt.plot(x, td3_return, 'salmon')
    plt.plot(x, smooth_data, 'r')
    plt.show()


if __name__ == "__main__":
    run_DQN()
    # run_TD3()