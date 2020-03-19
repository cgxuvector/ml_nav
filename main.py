from experiments import Experiment
from envs.LabEnv import RandomMaze
from agent.RandomAgent import RandomAgent
from agent.DQNAgent import DQNAgent
import argparse
import numpy as np


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="random", help="Type of the agent. (random, dqn, actor-critic)")
    parser.add_argument("--experiment_type", type=str, default="trn", help="Type of the experiment. "
                                                                           "(training or testing)")
    parser.add_argument("--rnd_seed", type=int, default=1234, help="random seed")
    parser.add_argument("--buffer_size", type=int, default=20_000, help="size of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the mini-batches")
    parser.add_argument("--max_total_time_steps", type=int, default=1_000_000, help="maximal number of time steps")
    parser.add_argument("--max_episode_time_steps", type=int, default=2_000, help="maximal time steps per episode")
    parser.add_argument("--use_replay_buffer", type=bool, default=True, help="whether use the replay buffer")

    parser.add_argument("--sampled_goal", type=int, default=5, help="number of sampled goals in each maze")
    parser.add_argument("--eval_frequency", type=int, default=100, help="frequency of evaluating the agent")

    parser.add_argument("--model_idx", type=str, default=None, help="model index")
    parser.add_argument("--save_dir", type=str, default=None, help="saving folder")
    return parser.parse_args()


if __name__ == '__main__':
    # load the input arguments
    inputs = parse_input()
    """ Set up the Deepmind environment"""
    # necessary observations (correct: this is the egocentric observations (following the counter clock direction))
    observation_list = ['RGB.LOOK_EAST',
                        'RGB.LOOK_NORTH_EAST',
                        'RGB.LOOK_NORTH',
                        'RGB.LOOK_NORTH_WEST',
                        'RGB.LOOK_WEST',
                        'RGB.LOOK_SOUTH_WEST',
                        'RGB.LOOK_SOUTH',
                        'RGB.LOOK_SOUTH_EAST',
                        'RGB.LOOK_RANDOM',
                        'DEBUG.POS.TRANS',
                        'DEBUG.POS.ROT',
                        'RGB.LOOK_TOP_DOWN']
    observation_width = 64
    observation_height = 64
    observation_fps = 60
    maze_size = [5, 7, 9]
    if inputs.experiment_type == 'trn':
        maze_seed = np.arange(15).tolist()
    else:
        maze_seed = np.arange(15, 20, 1).tolist()
    # create the environment
    my_lab = RandomMaze(observation_list, observation_width, observation_height, observation_fps)
    """ Set up the agent """
    if inputs.agent == 'random':
        my_agent = RandomAgent(my_lab.action_space, inputs.rnd_seed)
    elif inputs.agent == 'dqn':
        my_agent = DQNAgent(target_update_frequency=100,
                            policy_update_frequency=4,
                            soft_target_update=False,
                            dqn_mode="double",
                            gamma=0.99,
                            gradient_clip=False
                            )
    else:
        raise Exception(f"{inputs.agent} is not defined. Please try the valid agent (random, dqn, actor-critic)")
    """ Set up the experiment """
    my_experiment = Experiment.Experiment(
        env=my_lab,
        maze_list=maze_size,
        seed_list=maze_seed,
        agent=my_agent,
        buffer_size=inputs.buffer_size,
        batch_size=inputs.batch_size,
        max_time_steps=inputs.max_total_time_steps,
        max_time_steps_per_episode=inputs.max_episode_time_steps,
        use_replay=inputs.use_replay_buffer,
        sampled_goal=inputs.sampled_goal,
        gamma=0.99,
        start_train_step=100,
        model_name=inputs.model_idx,
        save_dir=inputs.save_dir
    )
    my_experiment.run()
