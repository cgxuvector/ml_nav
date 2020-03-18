from experiments import Experiment
from envs.LabEnv import RandomMaze
from agent.RandomAgent import RandomAgent
import argparse
import numpy as np

if __name__ == '__main__':
    """ Set up the Deepmind environment"""
    # necessary observations
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
    maze_size = [5]
    maze_seed = np.arange(20).tolist()
    # create the environment
    my_lab = RandomMaze(observation_list, observation_width, observation_height, observation_fps)
    """ Set up the agent """
    my_agent = RandomAgent(my_lab.action_space, 1234)
    """ Set up the experiment """
    my_experiment = Experiment.Experiment(
        env=my_lab,
        maze_list=maze_size,
        seed_list=maze_seed,
        agent=my_agent,
        buffer_size=10_000,
        max_time_steps=10_000,
        max_time_steps_per_episode=2_000,
        use_replay=True,
        gamma=0.99
    )
    my_experiment.run()
