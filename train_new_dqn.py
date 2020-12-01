from agent.DQNAgent_new import DQNAgent
from experiments.Experiment_new import DQNExperiment
from envs.LabEnvV2 import RandomMazeTileRaw
from envs.LabEnvV3 import RandomMazeTileRatio
from envs.LabEnvV4 import RandomMaze
import argparse
import IPython.terminal.debugger as Debug


DEFAULT_SIZE = [5, 7, 9, 11, 13, 15, 17, 19, 21]
DEFAULT_SEED = list(range(20))


def parse_input():
    parser = argparse.ArgumentParser()
    # set the agent
    parser.add_argument("--agent", type=str, default="random", help="Type of the agent (random, dqn, goal-dqn)")
    parser.add_argument("--dqn_mode", type=str, default="vanilla", help="Type of the DQN (vanilla, double)")
    # set the env params
    parser.add_argument("--env", type=str, default="DeepMind-discrete-raw", help="Version of domain")
    parser.add_argument("--maze_size", type=int, default=5, help="Maze size")
    parser.add_argument("--maze_seed", type=int, default=0, help="Maze seed")
    parser.add_argument("--fix_maze", action='store_true', default=False, help="Fix the maze")
    parser.add_argument("--fix_start", action='store_true', default=False, help="Fix the start position")
    parser.add_argument("--fix_goal", action='store_true', default=False, help="Fix the goal position")
    parser.add_argument("--use_true_state", action='store_true', default=False, help="Using true state flag")
    parser.add_argument("--use_obs", action='store_true', default=False, help="Using small observations flag")
    parser.add_argument("--use_goal", action='store_true', default=False, help="Using goal conditioned flag")
    parser.add_argument("--goal_dist", type=int, default=-1, help="Set distance between start and goal")
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
    parser.add_argument("--use_soft_target", action='store_true', default=False, help="Soft update flag")
    parser.add_argument("--polyak", type=float, default=0.95, help="Using soft update")
    # set the memory params
    parser.add_argument("--memory_size", type=int, default=20000, help="Memory size or replay buffer size")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of the mini-batch")
    # set RL params
    parser.add_argument("--gamma", type=float, default=0.99, help="Gamma")
    # set the saving params
    parser.add_argument("--model_name", type=str, default='null', help="model name")
    parser.add_argument("--save_dir", type=str, default='./', help="saving folder")
    # how to run DeepMind lab
    parser.add_argument("--env_render", type=str, default='software', help="Render: hardware for GPU machines and software for CPU machines")
    parser.add_argument("--env_run_mode", type=str, default='train', help="Mode of running the environment.")

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
    observation_width = 64
    observation_height = 64
    observation_fps = 60
    # set configurations
    configurations = {
        'width': str(observation_width),
        'height': str(observation_height),
        'fps': str(observation_fps)
    }

    # create the environment
    if input_args.env == 'DeepMind-discrete-raw':
        lab = RandomMazeTileRaw(level_name,
                                observation_list,
                                configurations,
                                args=inputs,
                                reward_type="sparse-1",
                                dist_epsilon=inputs.terminal_dist)
    elif input_args.env == 'DeepMind-discrete-ratio':
        lab = RandomMazeTileRatio(level_name,
                                  observation_list,
                                  configurations,
                                  args=inputs,
                                  reward_type="sparse-1",
                                  dist_epsilon=inputs.terminal_dist)
    elif input_args.env == 'DeepMind-continuous':
        lab = RandomMaze(level_name,
                         observation_list,
                         configurations,
                         args=inputs,
                         reward_type="sparse-1",
                         dist_epsilon=inputs.terminal_dist)
    else:
        raise Exception("Invalid environment name.")

    return lab


if __name__ == '__main__':
    # parse the input
    input_args = parse_input()

    # environment parameters
    env_params = {
        'env_name': input_args.env,
        'obs_dim': 3,
        'act_nun': 4,
        'run_eval_num': 10,
        'max_episode_time_steps': input_args.episode_time_steps
    }

    # agent parameters
    agent_params = {
        'dqn_mode': input_args.dqn_mode,
        'use_obs': input_args.use_obs,
        'gamma': input_args.gamma,
        'device': input_args.device,
        'use_soft_update': input_args.use_soft_update,
        'polyak': input_args.polyak
    }

    # training parameters
    train_params = {
        'memory_size': input_args.memory_size,
        'batch_size': input_args.batch_size,
        'lr': input_args.lr,
        'train_random_policy': input_args.train_random_policy,
        'total_time_steps': input_args.total_time_steps,
        'episode_time_steps': input_args.episode_time_steps,
        'start_train_step': input_args.start_train_step,
        'update_policy_freq': input_args.dqn_update_policy_freq,
        'update_target_freq': input_args.dqn_update_target_freq,
        'eval_policy_freq': input_args.eval_policy_freq,
        'model_name': input_args.model_name,
        'save_dir': input_args.save_dir
    }

    # create the environment
    my_env = make_env(input_args)
    my_env_test = make_env(input_args)

    # create the agent
    my_agent = DQNAgent(env_params, agent_params)

    # create training experiment
    my_experiment = DQNExperiment(my_env,
                                  my_env_test,
                                  my_agent,
                                  env_params,
                                  agent_params,
                                  train_params,
                                  input_args)
    # run the experiment
    my_experiment.run()
