import abc
import tqdm
import torch
import numpy as np
from utils.ExperienceReplay import ReplayBuffer
from utils.ml_schedule import LinearSchedule
from collections import defaultdict
from utils.mapper import RoughMap


class Experiment(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self):
        pass

    def get_action(self, time_t, obs_t):
        pass

    def step(self, action):
        pass

    def reset(self, set_new_maze):
        pass

    def update_models(self, time_t):
        pass

    def save_results(self, time_t):
        pass

    def eval_policy(self, time_t):
        pass


class DQNExperiment(Experiment):
    def __init__(self, env, env_test, agent, env_params, agent_params, train_params, args):
        # create the env
        self.env = env
        self.env_test = env_test
        self.env_params = env_params
        self.env_map = None
        self.env_map_test = None

        # create the agent
        self.agent = agent
        self.agent_params = agent_params

        # create the memory
        self.memory = ReplayBuffer(train_params['memory_size'])

        # create schedule
        self.schedule = LinearSchedule(1, 0.01, train_params['total_time_steps'] / 3)

        # create the training parameters
        self.train_params = train_params

        # extra arguments
        self.args = args

        # define training statistics
        self.train_returns = []
        self.eval_success = []

    def get_action(self, t, obs_t):
        # compute the epsilon for time step t
        self.agent.eps = self.schedule.get_value(t)
        # get the action
        return self.agent.get_action(obs_t)

    def step(self, action):
        if self.env_params['env_name'] == 'DeepMind-continuous':
            next_obs, reward, done, dist, trans, rots, trans_vel, rots_vel, _ = self.env.step(action)
            return np.array(next_obs), np.float(reward), np.bool(done), {'distance': dist,
                                                                         'trans': trans,
                                                                         'rots': rots,
                                                                         'trans_vel': trans_vel,
                                                                         'rots_vel': rots_vel}
        else:
            next_obs, reward, done, dist, trans, rots, _ = self.env.step(action)
            return np.array(next_obs), np.float(reward), np.bool(done), {'distance': dist,
                                                                         'trans': trans,
                                                                         'rots': rots}

    def reset(self, set_new_maze):
        # set maze configurations
        maze_configs = defaultdict(lambda: None)
        # set the maze size
        maze_size = self.args.maze_size
        maze_seed = self.args.maze_seed
        # set new maze flag
        if set_new_maze:
            # initialize the 2-D map
            self.env_map = RoughMap(maze_size, maze_seed, 3)
            # sample the start and goal position
            if not self.train_params['train_random_policy']:
                self.env_map.sample_start_goal_pos_with_random_dist(self.args.fix_start,
                                                                    self.args.fix_goal)
            else:
                self.env_map.sample_start_goal_pos_with_maximal_dist(self.args.fix_start,
                                                                     self.args.fix_goal,
                                                                     self.args.goal_dist)
            # initialize the maze 3D
            maze_configs["maze_name"] = f"maze_{maze_size}_{maze_seed}"  # string type name
            maze_configs["maze_size"] = [maze_size, maze_size]  # [int, int] list
            maze_configs["maze_seed"] = '1234'  # string type number
            maze_configs["maze_map_txt"] = "".join(self.env_map.map2d_txt)  # string type map
            maze_configs["maze_valid_pos"] = self.env_map.valid_pos  # list of valid positions
            # initialize the maze start and goal positions
            maze_configs["start_pos"] = self.env_map.init_pos + [0]
            maze_configs["goal_pos"] = self.env_map.goal_pos + [0]
            # initialize the update flag
            maze_configs["update"] = True  # update flag
        else:
            # sample another start and goal pair in the same maze
            if not self.train_params['train_random_policy']:
                self.env_map.sample_start_goal_pos_with_random_dist(self.args.fix_start,
                                                                    self.args.fix_goal)
            else:
                self.env_map.sample_start_goal_pos_with_maximal_dist(self.args.fix_start,
                                                                     self.args.fix_goal,
                                                                     self.args.goal_dist)
            maze_configs['start_pos'] = self.env_map.init_pos + [0]
            maze_configs['goal_pos'] = self.env_map.goal_pos + [0]
            maze_configs['maze_valid_pos'] = self.env_map.valid_pos
            maze_configs['update'] = False

        # obtain the state and goal observation
        if self.env_params['env_name'] == 'DeepMind-continuous':
            obs, g_obs, trans, rots, trans_vel, rots_vel = self.env.reset(maze_configs)
            return np.array(obs), np.array(g_obs), {'trans': trans,
                                                    'rots': rots,
                                                    'trans_vel': trans_vel,
                                                    'rots_vel': rots_vel}
        else:
            obs, g_obs, trans, rots = self.env.reset(maze_configs)
            return np.array(obs), np.array(g_obs), {'trans': trans,
                                                    'rots': rots}

    def update_models(self, t):
        # update the behavior policy
        if not np.mod(t, self.train_params['update_policy_freq']):
            # sample batch data
            batch_data = self.memory.sample_batch(self.train_params['batch_size'])
            # update the policy
            self.agent.update_behavior_policy(batch_data)

        # update the target policy
        if not np.mod(t, self.train_params['update_target_freq']):
            self.agent.update_target_policy()

    def eval_policy(self, t):
        # evaluate the policy
        if not np.mod(t, self.train_params['eval_policy_freq']):
            self.save_results(t)

            # reset the test environment
            obs, _, true_state = self.reset(set_new_maze=True)
            eps_old = self.agent.eps
            for r in range(10):
                success = 0
                with torch.no_grad():
                    for t in range(self.train_params['episode_time_steps']):
                        # get action
                        self.agent.eps = 0
                        action = self.agent.get_action(obs)
                        # step in the environment
                        next_obs, reward, done, true_state = self.step(action)
                        # check termination
                        if done:
                            # reset the environment
                            success = 1
                            obs, _, _ = self.reset(set_new_maze=False)
                            break
                        else:
                            # update to next obs
                            obs = next_obs
                # save the success
                self.eval_success.append(success)
            self.agent.eps = eps_old

    def save_results(self, t):
        # compute the model and the returns names
        return_file_name = self.args.save_dir + self.args.model_name + '_' + str(t) + '.npy'
        model_file_name = self.args.save_dir + self.args.model_name + '_' + str(t) + '.pt'
        # save the results
        np.save(return_file_name, self.train_returns)
        torch.save(self.agent.behavior_policy_net.state_dict(), model_file_name)

    def run(self):
        # initialize the environment
        obs, _, true_state = self.reset(set_new_maze=True)
        # define training indicators
        episode_t = 0
        rewards = []
        pbar = tqdm.trange(self.train_params['total_time_steps'])
        for t in pbar:
            # get action
            action = self.get_action(t, obs)
            # take one step
            next_obs, reward, done, true_state = self.step(action)
            # store the reward
            rewards.append(reward)
            # store the transition
            self.memory.add(obs, action, reward, next_obs, done)
            # check terminal
            if done or episode_t == self.train_params['episode_time_steps']:
                # compute the return
                G = 0
                for r in reversed(rewards):
                    G = r + self.agent.gamma * G
                self.train_returns.append(G)
                # compute the index of the episode
                episode_idx = len(self.train_returns)

                # print the info
                pbar.set_description(
                    f'Ep {episode_idx} | '
                    f'G {G:.3f} | '
                    f'Eval success {self.eval_success[-10:] if self.eval_success else 0: .2f} | '
                    f'Start {self.env_map.init_pos} | '
                    f'Goal {self.env_map.goal_pos}'
                )

                # reset the environment
                obs, _, true_state = self.reset(set_new_maze=False)
                rewards = []
                episode_t = 0
            else:
                episode_t += 1
                obs = next_obs

            # update the agent
            self.update_models(t)

            # evaluate the policy
            self.eval_policy(t)

        # save results
        self.save_results(self.train_params['total_time_steps'])




