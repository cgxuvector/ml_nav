import argparse
from envs.LabEnv import RandomMaze
from collections import namedtuple
from model.GoalDQN import DQN


def dqn_parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn_mode", type=str, default="vanilla", help="Mode of the DQN: vanilla or double")
    parser.add_argument("--clip_grad", type=bool, default=False, help="whether enable the grad clip")
    parser.add_argument("--save_name", type=str)
    return parser.parse_args()


def train_dqn():
    # input params
    input_args = dqn_parse_input()

    # observations
    observations = ['RGB.LOOK_EAST',
                    'RGB.LOOK_NORTH_EAST',
                    'RGB.LOOK_NORTH',
                    'RGB.LOOK_NORTH_WEST',
                    'RGB.LOOK_WEST',
                    'RGB.LOOK_SOUTH_WEST',
                    'RGB.LOOK_SOUTH',
                    'RGB.LOOK_SOUTH_EAST',
                    'RGB.LOOK_RANDOM',
                    'DEBUG.POS.TRANS',
                    'DEBUG.POS.ROT']

    # observation configuration
    obs_width = 64
    obs_height = 64
    obs_fps = 60

    # load the CarPole environment
    env = RandomMaze(observations, obs_width, obs_height, obs_fps, True, False)
    # define transitions
    transition_config = namedtuple("transition", ["state", "action", "next_state", "reward", "goal", "done"])
    # batch size and replay buffer
    batch_size = 64
    replay_buffer_size = 200_000

    # create a DQN
    myDQN = DQN(env=env,                                       # environment
                buffer_size=replay_buffer_size,                # replay buffer size
                batch_size=batch_size,                         # batch size
                max_time_steps=10_000,   # maximal time steps per epoch
                transition_config=transition_config,           # transition type
                learning_rate=1e-3,
                start_update_step=1000,
                target_update_frequency=10_000,
                policy_update_frequency=4,
                gamma=0.99,
                eps_start=1.0,
                dqn_mode=input_args.dqn_mode,
                gradient_clip=input_args.clip_grad,
                save_name=input_args.save_name
                )

    # train the DQN
    myDQN.train()


if __name__ == "__main__":
    train_dqn()