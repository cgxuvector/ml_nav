from test.Experiment import Experiment
from test.DQNAgent import DQNAgent
from test.RandomAgent import RandomAgent
from test.GoalDQNAgent import GoalDQNAgent
from envs.LabEnv import RandomMaze
import argparse
from collections import namedtuple
import sys
import IPython.terminal.debugger as Debug


DEFAULT_SIZE = [5, 7, 9, 11, 13]
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
    # set the running mode
    parser.add_argument("--run_num", type=int, default=1, help="Number of run for each experiment.")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed")
    # set the training mode
    parser.add_argument("--train_local_policy", action='store_true', default=False, help="Whether train a local policy.")
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
    parser.add_argument("--model_name", type=str, default=None, help="model name")
    parser.add_argument("--save_dir", type=str, default=None, help="saving folder")
    # add new strategy
    parser.add_argument("--use_rescale", action='store_true', default=False, help='whether rescale the value to [0,1]')
    parser.add_argument("--use_state_est", action='store_true', default=False, help='whether estimate the state')
    parser.add_argument("--alpha", type=float, default=1.0, help='hyperparameter for the two head case')
    return parser.parse_args()


if __name__ == '__main__':
    """ Load the input arguments """
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
    if inputs.use_small_obs:
        observation_width = 32
        observation_height = 32
    else:
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
    my_lab = RandomMaze(observation_list, observation_width, observation_height, observation_fps, reward_type="sparse-1")

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
                                device=inputs.device,
                                use_small_obs=inputs.use_small_obs
                            )
    else:
        raise Exception(f"{inputs.agent} is not defined. Please try the valid agent (random, dqn, actor-critic)")

    """ Set up the experiment """
    transition = namedtuple("transition", ["state", "action", "reward", "next_state", "done"]) if not inputs.use_goal \
        else namedtuple("transition", ["state", "action", "reward", "next_state", "goal", "done"])
    my_experiment = Experiment(
        env=my_lab,
        maze_list=maze_size,
        seed_list=maze_seed,
        fix_maze=inputs.fix_maze,
        fix_start=inputs.fix_start,
        fix_goal=inputs.fix_goal,
        sampled_goal=inputs.sampled_goal_num,
        agent=my_agent,
        buffer_size=inputs.memory_size,
        batch_size=inputs.batch_size,
        max_time_steps=inputs.total_time_steps,
        episode_time_steps=inputs.episode_time_steps,
        use_replay=inputs.use_memory,
        gamma=inputs.gamma,
        start_train_step=inputs.start_train_steps,
        model_name=inputs.model_idx,
        save_dir=inputs.save_dir,
        train_episode_num=inputs.train_episode_num,
        # whether use goal-conditioned strategy
        use_goal=inputs.use_goal,
        transition=transition,
        goal_dist=inputs.goal_dist,
        random_seed=inputs.rnd_seed
    )
    # run the experiments
    my_experiment.random_goal_conditioned_her_run()



