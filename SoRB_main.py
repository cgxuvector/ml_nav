"""
    Implementation of training SoRB. Actually, the training only contains train a goal-conditioned DQN agent.
    Currently, I don't use distributional RL. Just use the same model but without discounted factor
"""
from baselines.SoRB_agent import GoalDQNAgent
from baselines.SoRB_experiment import Experiment
from envs.LabEnvV2 import RandomMazeTileRaw
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
    parser.add_argument("--distance_list", type=str, default="1", help="Maze distance list")
    parser.add_argument("--fix_maze", type=str, default="True", help="Fix the maze")
    parser.add_argument("--fix_start", type=str, default="True", help="Fix the start position")
    parser.add_argument("--fix_goal", type=str, default="True", help="Fix the goal position")
    parser.add_argument("--decal_freq", type=float, default=0.001, help="Wall decorator frequency")
    parser.add_argument("--use_true_state", type=str, default="True", help="Using true state flag")
    parser.add_argument("--use_small_obs", type=str, default='False', help="Using small observations flag")
    parser.add_argument("--use_goal", type=str, default="False", help="Using goal conditioned flag")
    parser.add_argument("--goal_dist", type=int, default=-1, help="Set distance between start and goal")
    parser.add_argument("--use_imagine", type=str, default="False", help="Using imagination of goal")
    # set the running mode
    parser.add_argument("--run_num", type=int, default=1, help="Number of run for each experiment.")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed")
    # set the training mode
    parser.add_argument("--train_local_policy", type=str, default="False", help="Whether train a local policy.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--start_train_step", type=int, default=1000, help="Start training time step")
    parser.add_argument("--sampled_goal_num", type=int, default=10, help="Number of sampled start and goal positions.")
    parser.add_argument("--train_episode_num", type=int, default=10, help="Number of training epochs for each sample.")
    parser.add_argument("--total_time_steps", type=int, default=50000, help="Total time steps")
    parser.add_argument("--episode_time_steps", type=int, default=100, help="Time steps per episode")
    parser.add_argument("--eval_policy_freq", type=int, default=1000, help="Evaluate the current learned policy frequency")
    parser.add_argument("--dqn_update_target_freq", type=int, default=2000, help="Frequency of updating the target")
    parser.add_argument("--dqn_update_policy_freq", type=int, default=10, help="Frequency of updating the policy")
    parser.add_argument("--soft_target_update", type=str, default="True", help="Soft update flag")
    parser.add_argument("--dqn_gradient_clip", type=str, default="True", help="Clip the gradient flag")
    parser.add_argument("--mix_maze", type=str, default="False", help="If set true, "
                                                                      "then the training mazes are mixed size")
    parser.add_argument("--fold_k", type=int, default=-1, help="Cross validation")
    parser.add_argument("--trn_num_each_maze", type=int, default=1, help="Number of the training mazes for each size")
    # set the memory params
    parser.add_argument("--memory_size", type=int, default=1000, help="Memory size or replay buffer size")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of the mini-batch")
    parser.add_argument("--use_memory", type=str, default="True", help="If true, use the memory")
    parser.add_argument("--use_her", type=str, default="False", help="If true, use the Hindsight Experience Replay")
    parser.add_argument("--future_k", type=int, default=4, help="Number of sampling future states")
    # set RL params
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma")
    # set the saving params
    parser.add_argument("--model_idx", type=str, default=None, help="model index")
    parser.add_argument("--save_dir", type=str, default=None, help="saving folder")

    # flag for SoRB
    parser.add_argument("--distributional_rl", type=str, default='False', help="Whether use distributional RL or not")
    parser.add_argument("--support_atoms", type=int, default='51', help="Number of the support atoms")
    parser.add_argument("--run_mode", type=str, default='trn', help="Whether train or evaluate the SoRB")
    parser.add_argument("--max_dist", type=int, default=1, help="Max distance")
    parser.add_argument("--gpu_acc", type=str, default='False', help="Whether use GPU to boost imagine rendering")
    parser.add_argument("--use_rescale", type=str, default='False', help="Whether use pixel value between [0, 1]")

    # evaluation
    parser.add_argument("--model_maze_size", type=int, default=13)
    parser.add_argument("--model_dist", type=int, default=1)
    parser.add_argument("--model_seed", type=int, default=0)

    return parser.parse_args()


# convert the True/False from str to bool
def strTobool(inputs):
    # set params of mazes
    inputs.fix_maze = True if inputs.fix_maze == "True" else False
    inputs.fix_start = True if inputs.fix_start == "True" else False
    inputs.fix_goal = True if inputs.fix_goal == "True" else False
    inputs.use_small_obs = True if inputs.use_small_obs == "True" else False
    inputs.use_true_state = True if inputs.use_true_state == "True" else False
    inputs.use_goal = True if inputs.use_goal == "True" else False
    inputs.use_imagine = True if inputs.use_imagine == "True" else False
    # set params of training
    inputs.train_local_policy = True if inputs.train_local_policy == "True" else False
    inputs.use_memory = True if inputs.use_memory == "True" else False
    inputs.soft_target_update = True if inputs.soft_target_update == 'True' else False
    inputs.dqn_gradient_clip = True if inputs.dqn_gradient_clip == 'True' else False
    # set params of HER
    inputs.use_her = True if inputs.use_her == "True" else False
    # use mixed mazes as training
    inputs.mix_maze = True if inputs.mix_maze == "True" else False
    # use distributional rl
    inputs.distributional_rl = True if inputs.distributional_rl == "True" else False
    # use GPU to accelerate
    inputs.gpu_acc = True if inputs.gpu_acc == "True" else False
    # use rescale value
    inputs.use_rescale = True if inputs.use_rescale == "True" else False
    
    return inputs


# make the environment
def make_env(inputs):
    # set level name
    level_name = 'nav_random_maze_tile_bsp'
    # necessary observations (correct: this is the egocentric observations (following the counter clock direction))
    observation_list = [
        'RGB.LOOK_RANDOM_PANORAMA_VIEW',
        'RGB.LOOK_TOP_DOWN_VIEW'
    ]
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
    if len(inputs.maze_size_list) == 1 or len(inputs.maze_size_list) == 2:
        maze_size = [int(inputs.maze_size_list)]
    else:
        maze_size = [int(s) for s in inputs.maze_size_list.split(",")]
    if len(inputs.maze_seed_list) == 1 or len(inputs.maze_seed_list) == 2:
        maze_seed = [int(inputs.maze_seed_list)]
    else:
        maze_seed = [int(s) for s in inputs.maze_seed_list.split(",")]
    if len(inputs.distance_list) == 1 or len(inputs.distance_list) == 2:
        maze_dist = [int(inputs.distance_list)]
    else:
        maze_dist = [int(d) for d in inputs.distance_list.split(",")] 
    assert set(maze_size) <= set(DEFAULT_SIZE), f"Input contains invalid maze size. Expect a subset of {DEFAULT_SIZE}, " \
                                                f"but get {maze_size}."
    assert set(maze_seed) <= set(DEFAULT_SEED), f"Input contains invalid maze seed. Expect a subset of {DEFAULT_SEED}, " \
                                                f"but get {maze_seed}."
    # create the environment
    lab = RandomMazeTileRaw(level_name,
                            observation_list,
                            configurations,
                            use_true_state=inputs.use_true_state,
                            reward_type="sparse-1",
                            dist_epsilon=1e-3)
    return lab, maze_size, maze_seed, maze_dist


# make the agent
def make_agent(inputs):
    # two mode: vanilla goal-conditioned DQN and goal-conditioned DQN with distributional RL
    if inputs.distributional_rl:
        agent = GoalDQNAgent(dqn_mode=inputs.dqn_mode,
                             target_update_frequency=inputs.dqn_update_target_freq,
                             policy_update_frequency=inputs.dqn_update_policy_freq,
                             use_small_obs=inputs.use_small_obs,
                             use_true_state=inputs.use_true_state,
                             use_target_soft_update=inputs.soft_target_update,
                             use_gradient_clip=inputs.dqn_gradient_clip,
                             gamma=inputs.gamma,
                             device=inputs.device,
                             use_distributional=True,
                             support_atoms=inputs.support_atoms,
                             batch_size=inputs.batch_size,
                             use_rescale=inputs.use_rescale
                             )
    else:
        agent = GoalDQNAgent(dqn_mode=inputs.dqn_mode,
                             target_update_frequency=inputs.dqn_update_target_freq,
                             policy_update_frequency=inputs.dqn_update_policy_freq,
                             use_small_obs=inputs.use_small_obs,
                             use_true_state=inputs.use_true_state,
                             use_target_soft_update=inputs.soft_target_update,
                             use_gradient_clip=inputs.dqn_gradient_clip,
                             gamma=inputs.gamma,
                             device=inputs.device,
                             use_distributional=False,
                             use_rescale=inputs.use_rescale
                             )
    return agent


# run experiment
def run_experiment(inputs):
    # create the environment
    my_lab, size, seed, dist= make_env(inputs)
    # create the agent
    my_agent = make_agent(inputs)
    # create the transition
    transition = namedtuple("transition", ["state", "action", "reward", "next_state", "done"]) if not inputs.use_goal \
        else namedtuple("transition", ["state", "action", "reward", "next_state", "goal", "done"])
    # create the experiment
    my_experiment = Experiment(
        env=my_lab,
        agent=my_agent,
        maze_list=size,
        seed_list=seed,
        dist_list=dist,
        decal_freq=inputs.decal_freq,
        fix_maze=inputs.fix_maze,
        fix_start=inputs.fix_start,
        fix_goal=inputs.fix_goal,
        use_goal=inputs.use_goal,
        goal_dist=inputs.goal_dist,
        max_dist=inputs.max_dist,
        use_true_state=inputs.use_true_state,
        sample_start_goal_num=inputs.sampled_goal_num,
        train_episode_num=inputs.train_episode_num,
        start_train_step=inputs.start_train_step,
        max_time_steps=inputs.total_time_steps,
        episode_time_steps=inputs.episode_time_steps,
        use_replay=inputs.use_memory,
        use_her=inputs.use_her,
        future_k=inputs.future_k,
        buffer_size=inputs.memory_size,
        transition=transition,
        batch_size=inputs.batch_size,
        gamma=inputs.gamma,
        save_dir=inputs.save_dir,
        model_name=inputs.model_idx,
        device=inputs.device,
        use_rescale=inputs.use_rescale,
        eval_policy_freq=inputs.eval_policy_freq,
        args=inputs
    )
    # run the experiments
    if inputs.run_mode == 'trn':
        my_experiment.train_local_goal_conditioned_dqn()
    elif inputs.run_mode == 'trn-her':
        my_experiment.train_local_goal_conditioned_dqn_with_her()
    elif inputs.run_mode == 'tst-dist':
        my_experiment.test_distance_prediction()
    elif inputs.run_mode == 'sorb':
        my_experiment.run_SoRB()
    else:
        raise Exception(f"Invalid experiment running mode. Expect one from (trn, trn-her, tst-dist, or sorb), but get"
                        f"{inputs.run_mode}")


if __name__ == '__main__':
    # load the input parameters
    user_inputs = parse_input()
    user_inputs = strTobool(user_inputs)

    # user input model index
    input_model_idx = user_inputs.model_idx
    input_save_dir = user_inputs.save_dir

    # total seed list
    total_seed_list = user_inputs.maze_seed_list.split(',')

    input_maze_size_list = user_inputs.maze_size_list.split(',') 
    for s in input_maze_size_list:
        # set the random seed for reproduce
        random.seed(user_inputs.random_seed)
        np.random.seed(user_inputs.random_seed)
        torch.manual_seed(user_inputs.random_seed)

        # set the maze size list
        user_inputs.maze_size_list = s

        # print info
        print(f"Run the experiment with random seed = {user_inputs.random_seed} using mazes size {s} and seed {user_inputs.maze_seed_list}")

        # update the save directory
        user_inputs.save_dir = input_save_dir + '/sorb/' + f'{s}x{s}' + f'/{user_inputs.random_seed}'
        user_inputs.model_idx = input_model_idx + f'_{s}x{s}_obs_sorb_her'

        # check the directory to store the results
        #if not os.path.exists(user_inputs.save_dir):
        #    os.makedirs(user_inputs.save_dir)

        # run the experiments
        run_experiment(user_inputs) 
