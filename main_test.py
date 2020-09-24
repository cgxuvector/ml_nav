from test.Experiment import Experiment
from test.GoalDQNAgent import GoalDQNAgent
from test.DQNAgent import DQNAgent
from envs.LabEnvV4 import RandomMaze
from collections import namedtuple
import argparse
import torch
import random
import os
import numpy as np
import IPython.terminal.debugger as Debug


DEFAULT_SIZE = [5, 7, 9, 11, 13, 15, 17, 19, 21]
DEFAULT_SEED = list(range(20))


def parse_input():
    parser = argparse.ArgumentParser()
    # set the agent
    parser.add_argument("--agent", type=str, default="random", help="Type of the agent (random, dqn, goal-dqn)")
    parser.add_argument("--dqn_mode", type=str, default="vanilla", help="Type of the DQN (vanilla, double)")
    # set the env params
    parser.add_argument("--maze_size_list", type=str, default="5", help="Maze size list")
    parser.add_argument("--maze_seed_list", type=str, default="0", help="Maze seed list")
    parser.add_argument("--fix_maze", action='store_true', default=False, help="Fix the maze")
    parser.add_argument("--fix_start", action='store_true', default=False, help="Fix the start position")
    parser.add_argument("--fix_goal", action='store_true', default=False, help="Fix the goal position")
    parser.add_argument("--decal_freq", type=float, default=0.001, help="Wall decorator frequency")
    parser.add_argument("--use_true_state", action='store_true', default=False, help="Using true state flag")
    parser.add_argument("--use_small_obs", action='store_true', default=False, help="Using small observations flag")
    parser.add_argument("--use_goal", action='store_true', default=False, help="Using goal conditioned flag")
    parser.add_argument("--goal_dist", type=int, default=-1, help="Set distance between start and goal")
    parser.add_argument("--use_imagine", type=float, default=0, help="Proportion of relabeled imagination goal")
    parser.add_argument("--terminal_dist", type=float, default=4.0, help="Termination distance for one episode")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed")
    # set the training mode
    parser.add_argument("--train_random_policy", action='store_true', default=False, help="Whether train a local policy.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--start_train_step", type=int, default=1000, help="Start training time step")
    parser.add_argument("--sampled_goal_num", type=int, default=10, help="Number of sampled start and goal positions.")
    parser.add_argument("--train_episode_num", type=int, default=10, help="Number of training epochs for each sample.")
    parser.add_argument("--total_time_steps", type=int, default=50000, help="Total time steps")
    parser.add_argument("--episode_time_steps", type=int, default=100, help="Time steps per episode")
    parser.add_argument("--eval_policy_freq", type=int, default=10, help="Evaluate the current learned policy frequency")
    parser.add_argument("--dqn_update_target_freq", type=int, default=1000, help="Frequency of updating the target")
    parser.add_argument("--dqn_update_policy_freq", type=int, default=4, help="Frequency of updating the policy")
    parser.add_argument("--soft_target_update", action='store_true', default=False, help="Soft update flag")
    parser.add_argument("--dqn_gradient_clip", action='store_true', default=False, help="Clip the gradient flag")
    parser.add_argument("--train_maze_num", type=int, default=1, help="Number of the training mazes")
    # set the memory params
    parser.add_argument("--memory_size", type=int, default=20000, help="Memory size or replay buffer size")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of the mini-batch")
    parser.add_argument("--use_memory", action='store_true', default=False, help="If true, use the memory")
    parser.add_argument("--use_her", action='store_true', default=False, help="If true, use the Hindsight Experience Replay")
    parser.add_argument("--future_k", type=int, default=4, help="Number of sampling future states")
    # set RL params
    parser.add_argument("--gamma", type=float, default=0.99, help="Gamma")
    # set the saving params
    parser.add_argument("--model_name", type=str, default='null', help="model name")
    parser.add_argument("--save_dir", type=str, default='./', help="saving folder")
    # add new strategy
    parser.add_argument("--use_rescale", action='store_true', default=False, help='whether rescale the value to [0,1]')
    parser.add_argument("--use_state_est", action='store_true', default=False, help='whether estimate the state')
    parser.add_argument("--alpha", type=float, default=1.0, help='hyperparameter for the two head case')

    # parameters related to new training paradigm
    parser.add_argument("--start_radius", type=float, default=0, help="Radius for sampling the start position.")

    return parser.parse_args()


# make the environment
def make_env(inputs):
    # set level name
    level_name = 'nav_random_maze_tile_bsp'
    # egocentric panoramic observation
    observation_list = [
        'RGB.LOOK_RANDOM_PANORAMA_VIEW',
        'RGB.LOOK_TOP_DOWN_VIEW',
        'RGB.LOOK_PANORAMA_VIEW',
        'DEBUG.POS.TRANS',
        'DEBUG.POS.ROT',
        'VEL.TRANS',
        'VEL.ROT'
    ]
    # set observation size
    if inputs.use_small_obs:
        observation_width = 32
        observation_height = 32
    else:
        observation_width = 64
        observation_height = 64
    observation_fps = 60
    # set configurations
    configurations = {
        'width': str(observation_width),
        'height': str(observation_height),
        'fps': str(observation_fps)
    }
    # set the mazes
    maze_size_list = [int(m_s) for m_s in inputs.maze_size_list.split(",")]
    maze_seed_list = [int(m_s) for m_s in inputs.maze_seed_list.split(",")]
    assert set(maze_size_list) <= set(DEFAULT_SIZE), f"Containing invalid maze size. Expect a subset of" \
                                                     f" {DEFAULT_SIZE}, but get {maze_size_list}."
    assert set(maze_seed_list) <= set(DEFAULT_SEED), f"Containing invalid maze seed. Expect a subset of" \
                                                     f" {DEFAULT_SEED}, but get {maze_seed_list}."
    # create the environment
    lab = RandomMaze(level_name,
                     observation_list,
                     configurations,
                     args=inputs,
                     use_true_state=inputs.use_true_state,
                     reward_type="sparse-1",
                     dist_epsilon=inputs.terminal_dist)

    return lab, maze_size_list, maze_seed_list


# make the agent
def make_agent(inputs):
    if inputs.agent == 'dqn':
        agent = DQNAgent(dqn_mode=inputs.dqn_mode,
                         target_update_frequency=inputs.dqn_update_target_freq,
                         policy_update_frequency=inputs.dqn_update_policy_freq,
                         use_small_obs=inputs.use_small_obs,
                         use_true_state=inputs.use_true_state,
                         use_target_soft_update=inputs.soft_target_update,
                         use_gradient_clip=inputs.dqn_gradient_clip,
                         gamma=inputs.gamma,
                         device=inputs.device,
                         )
    elif inputs.agent == 'goal-dqn':
        agent = GoalDQNAgent(dqn_mode=inputs.dqn_mode,
                             target_update_frequency=inputs.dqn_update_target_freq,
                             policy_update_frequency=inputs.dqn_update_policy_freq,
                             use_small_obs=inputs.use_small_obs,
                             use_true_state=inputs.use_true_state,
                             use_target_soft_update=inputs.soft_target_update,
                             use_gradient_clip=inputs.dqn_gradient_clip,
                             gamma=inputs.gamma,
                             device=inputs.device,
                             use_rescale=inputs.use_rescale,
                             use_state_est=inputs.use_state_est,
                             alpha=inputs.alpha,
                             )
    else:
        raise Exception(f"{inputs.agent} is not defined. Please try the valid agent (random, dqn, actor-critic)")

    return agent


# run experiment
def run_experiment(inputs):
    # create the environment
    my_lab, size_list, seed_list = make_env(inputs)
    # create the agent
    my_agent = make_agent(inputs)
    # create the transition
    if not inputs.use_goal:
        transition = namedtuple("transition", ["state", "action", "reward", "next_state", "done"])
    else:
        transition = namedtuple("transition", ["state", "action", "reward", "next_state", "goal", "done"])
    # create the experiment
    my_experiment = Experiment(
        env=my_lab,
        agent=my_agent,
        maze_list=size_list,
        seed_list=seed_list,
        decal_freq=inputs.decal_freq,
        fix_maze=inputs.fix_maze,
        fix_start=inputs.fix_start,
        fix_goal=inputs.fix_goal,
        use_goal=inputs.use_goal,
        goal_dist=inputs.goal_dist,
        use_true_state=inputs.use_true_state,
        train_random_policy=inputs.train_random_policy,
        sample_start_goal_num=inputs.sampled_goal_num,
        train_episode_num=inputs.train_episode_num,
        start_train_step=inputs.start_train_step,
        max_time_steps=inputs.total_time_steps,
        episode_time_steps=inputs.episode_time_steps,
        eval_policy_freq=inputs.eval_policy_freq,
        use_replay=inputs.use_memory,
        use_her=inputs.use_her,
        future_k=inputs.future_k,
        buffer_size=inputs.memory_size,
        transition=transition,
        learning_rate=inputs.lr,
        batch_size=inputs.batch_size,
        gamma=inputs.gamma,
        save_dir=inputs.save_dir,
        model_name=inputs.model_name,
        use_imagine=inputs.use_imagine,
        device=inputs.device,
        use_state_est=inputs.use_state_est
    )

    # run the experiments
    if inputs.use_goal:
        # train a global goal-conditioned policy
        if not inputs.train_random_policy:
            # my_experiment.run_goal_dqn()
            my_experiment.run_goal_dqn_map_guide_explore()
        else:
            if not inputs.use_her:
                my_experiment.run_random_local_goal_dqn()
            else:
                my_experiment.run_random_goal_dqn_her()
    else:
        # train a vanilla policy
        my_experiment.run_dqn()


if __name__ == '__main__':
    # load the input parameters
    user_inputs = parse_input()

    # user input model index
    input_model_name = user_inputs.model_name
    input_save_dir = user_inputs.save_dir

    # total seed list
    total_seed_list = user_inputs.maze_seed_list.split(',')

    """ Experiment Paradigm
        - One run for different sizes of mazes
        - Sample #train_maze_num mazes for training
    """
    # run experiments
    input_maze_size_list_init = user_inputs.maze_size_list.split(',')  # obtain all maze sizes
    input_maze_seed_list_init = user_inputs.maze_seed_list.split(',')  # obtain all maze seeds
    # loop maze sizes
    for s in input_maze_size_list_init:
        # set random seed for reproduce
        random.seed(user_inputs.random_seed)
        np.random.seed(user_inputs.random_seed)
        torch.manual_seed(user_inputs.random_seed)

        # set the training maze size list
        user_inputs.maze_size_list = s

        # set the training maze seed list
        input_maze_seed_shuffle = input_maze_seed_list_init.copy()
        np.random.shuffle(input_maze_seed_shuffle)
        if len(input_maze_seed_list_init) > user_inputs.train_maze_num:
            user_inputs.maze_seed_list = ','.join(random.sample(input_maze_seed_shuffle, user_inputs.train_maze_num))
        else:
            user_inputs.maze_seed_list = ','.join(input_maze_seed_shuffle)

        # print info
        print(f"Run the experiment with random seed = {user_inputs.random_seed} using mazes size {s} and"
              f" seed {user_inputs.maze_seed_list}")

        # update the save directory
        if user_inputs.train_maze_num > 1:
            user_inputs.save_dir = input_save_dir + '/few_shot/' + f'{s}-{user_inputs.goal_dist}' + f'/{user_inputs.random_seed}'
        else:
            user_inputs.save_dir = input_save_dir + '/one_shot/' + f'{s}-{user_inputs.goal_dist}' + f'/{user_inputs.random_seed}'
        user_inputs.model_name = input_model_name + f'_{s}x{s}_obs_dist_{user_inputs.goal_dist}'

        # check the directory to store the results
        if not os.path.exists(user_inputs.save_dir):
            os.makedirs(user_inputs.save_dir)

        # run experiments
        run_experiment(user_inputs)
