from test.Experiment import Experiment
from test.DQNAgent import DQNAgent
from test.RandomAgent import RandomAgent
from test.GoalDQNAgent import GoalDQNAgent
from envs.LabEnv import RandomMaze
import argparse
from collections import namedtuple
import sys


DEFAULT_SIZE = [5, 7, 9, 11, 13]
DEFAULT_SEED = list(range(20))


def parse_input():
    parser = argparse.ArgumentParser()
    # set the agent
    parser.add_argument("--agent", type=str, default="random", help="Type of the agent. (random, dqn)")
    parser.add_argument("--dqn_mode", type=str, default="vanilla", help="Type of the DQN. (vanilla, double)")
    # set the env params
    parser.add_argument("--maze_size_list", type=str, default="5", help="Maze size list. [5, 7, 9, 11, 13]")
    parser.add_argument("--maze_seed_list", type=str, default="0", help="Maze seed list. [0, 1, 2 .. ,19]")
    parser.add_argument("--fix_maze", type=bool, default=True, help="Fix the maze.")
    parser.add_argument("--fix_start", type=bool, default=True, help="Fix the initial position.")
    parser.add_argument("--fix_goal", type=bool, default=True, help="Fix the goal position.")
    parser.add_argument("--train_episode_num", type=int, default=100, help="Number of episode to train for each sampling")
    parser.add_argument("--sampled_goal", type=int, default=10, help="Number of sampled initial and goal positions.")
    # set the running mode
    parser.add_argument("--rnd_seed", type=int, default=1234, help="Random seed.")
    # set the training mode
    parser.add_argument("--device", type=str, default="cpu", help="Device to use.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--start_train_steps", type=int, default=100, help="Start training time steps.")
    parser.add_argument("--total_time_steps", type=int, default=1000000, help="Total time steps.")
    parser.add_argument("--episode_time_steps", type=int, default=2000, help="Time steps per episode.")
    parser.add_argument("--dqn_update_target_net", type=int, default=100, help="Frequency of updating the target.")
    parser.add_argument("--dqn_update_policy_net", type=int, default=4, help="Frequency of updating the policy.")
    parser.add_argument("--soft_target_update_tau", type=float, default=0.05, help="Soft update params.")
    parser.add_argument("--dqn_gradient_clip", type=bool, default=False, help="If true, clip the gradient.")
    # set the memory params
    parser.add_argument("--memory_size", type=int, default=20000, help="Memory size or replay buffer size.")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of the mini-batch.")
    parser.add_argument("--use_memory", type=bool, default=True, help="If true, use the memory.")
    # set RL params
    parser.add_argument("--gamma", type=float, default=0.995, help="Gamma")
    # set the saving params
    parser.add_argument("--model_idx", type=str, default=None, help="model index")
    parser.add_argument("--save_dir", type=str, default=None, help="saving folder")
    # set goal-conditioned
    parser.add_argument("--use_goal", type=str, default=None, help="whether using goal conditioned strategy")
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
    if len(inputs.maze_size_list) == 1:
        maze_size = [int(inputs.maze_size_list)]
    else:
        maze_size = [int(s) for s in inputs.maze_size_list.split(",")]
    if len(inputs.maze_seed_list) == 1:
        maze_seed = [int(inputs.maze_seed_list)]
    else:
        maze_seed = [int(s) for s in inputs.maze_seed_list.split(",")]
    assert set(maze_size) <= set(DEFAULT_SIZE), f"Input contains invalid maze size. Expect a subset of {DEFAULT_SIZE}, " \
                                                f"but get {maze_size}."
    assert set(maze_seed) <= set(DEFAULT_SEED), f"Input contains invalid maze seed. Expect a subset of {DEFAULT_SEED}, " \
                                                f"but get {maze_seed}."
    # create the environment
    my_lab = RandomMaze(observation_list, observation_width, observation_height, observation_fps)
    """ Set up the agent """
    if inputs.agent == 'random':
        my_agent = RandomAgent(my_lab.action_space, inputs.rnd_seed)
    elif inputs.agent == 'dqn':
        my_agent = DQNAgent(target_update_frequency=inputs.dqn_update_target_net,
                            policy_update_frequency=inputs.dqn_update_policy_net,
                            soft_target_update_tau=inputs.soft_target_update_tau,
                            dqn_mode=inputs.dqn_mode,
                            gamma=0.99,
                            gradient_clip=inputs.dqn_gradient_clip,
                            device=inputs.device
                            )
    elif inputs.agent == 'goal-dqn':
        my_agent = GoalDQNAgent(target_update_frequency=inputs.dqn_update_target_net,
                                policy_update_frequency=inputs.dqn_update_policy_net,
                                soft_target_update_tau=inputs.soft_target_update_tau,
                                dqn_mode=inputs.dqn_mode,
                                gamma=0.99,
                                gradient_clip=inputs.dqn_gradient_clip,
                                device=inputs.device
                            )
    else:
        raise Exception(f"{inputs.agent} is not defined. Please try the valid agent (random, dqn, actor-critic)")
    """ Set up the experiment """
    my_experiment = Experiment(
        env=my_lab,
        maze_list=maze_size,
        seed_list=maze_seed,
        fix_maze=inputs.fix_maze,
        fix_start=inputs.fix_start,
        fix_goal=inputs.fix_goal,
        sampled_goal=inputs.sampled_goal,
        agent=my_agent,
        buffer_size=inputs.memory_size,
        batch_size=inputs.batch_size,
        max_time_steps=inputs.total_time_steps,
        max_time_steps_per_episode=inputs.episode_time_steps,
        use_replay=inputs.use_memory,
        gamma=inputs.gamma,
        start_train_step=inputs.start_train_steps,
        model_name=inputs.model_idx,
        save_dir=inputs.save_dir,
        train_episode_num=inputs.train_episode_num,
        # whether use goal-conditioned strategy
        use_goal=True,
        transition=namedtuple("transition", ["state", "action", "reward", "next_state", "goal", "done"])
    )
    my_experiment.run()



