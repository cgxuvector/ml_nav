""" This script is a demo to control agent in a specific static maze environment in DeepMind Lab.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import random
from model import VAE
from test.DQNAgent import DQNAgent
from test.GoalDQNAgent import GoalDQNAgent
from utils import mapper
from collections import defaultdict
from envs.LabEnvV2 import RandomMazeTileRaw
import numpy as np
import matplotlib.pyplot as plt
import IPython.terminal.debugger as Debug

ACTION_LIST = ['up', 'down', 'left', 'right']


class VisualPolicy(object):
    def __init__(self, env, agent, size, set_goal, fix_start, fix_goal, g_dist, use_obs, use_imagine):
        self.env = env
        self.env_map = None
        self.agent = agent
        self.maze_size = size
        self.maze_seed = 0
        self.maze_size_list = [size]
        self.maze_seed_list = [0]
        self.fix_start = fix_start
        self.fix_goal = fix_goal
        self.theme_list = ['MISHMASH']
        self.decal_list = [0.001]
        self.use_goal = set_goal
        self.use_obs = use_obs
        self.use_imagine = use_imagine
        self.goal_dist = g_dist
        self.gamma = 0.99
        self.orientations = [torch.tensor([0, 0, 0, 0, 1, 0, 0, 0]),
                             torch.tensor([0, 0, 1, 0, 0, 0, 0, 0]),
                             torch.tensor([0, 1, 0, 0, 0, 0, 0, 0]),
                             torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]),
                             torch.tensor([0, 0, 0, 1, 0, 0, 0, 0]),
                             torch.tensor([0, 0, 0, 0, 0, 1, 0, 0]),
                             torch.tensor([0, 0, 0, 0, 0, 0, 1, 0]),
                             torch.tensor([0, 0, 0, 0, 0, 0, 0, 1])]
        # load the vae model
        self.cvae = VAE.CVAE(64, use_small_obs=True)
        self.cvae.load_state_dict(torch.load("./results/vae/model/small_obs_L64_B8.pt", map_location='cpu'))
        self.cvae.eval()

    def run_fixed_start_goal_pos(self):
        # init the environment
        self.fix_start = True
        self.fix_goal = True
        state, goal, _, _ = self.update_map2d_and_maze3d(set_new_maze=True)

        # episodes
        episode_num = 10
        states = [state]
        rewards = []
        actions = []
        for t in range(episode_num):
            # get one action
            if self.use_goal:
                action = self.agent.get_action(state, goal, 0)
            else:
                action = self.agent.get_action(state, 0)

            # step in the environment
            next_state, reward, done, dist, trans, _, _ = my_lab.step(action)

            # save the results
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)

            # print the steps
            print(f"Current state = {trans}, Action = {ACTION_LIST[action]}, Next state = {trans}")

            # check terminal
            if done:
                break
            else:
                state = next_state

        # compute the discounted return
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
        print("Discounted return of the init state = {}".format(G))

    def run_random_start_goal_pos(self):
        # init the environment
        self.update_map2d_and_maze3d(set_new_maze=True)

        # testing running number
        run_num = 1000
        # success counter
        success_count = 0
        # start testing
        for r in range(run_num):
            # sample a random start and goal position
            self.fix_start = False
            self.fix_goal = False
            state, goal, _, _ = self.update_map2d_and_maze3d(set_new_maze=False)
            print("Run idx = {}, Init = {}, Goal = {}".format(r, state, goal))
            # set the maximal steps
            episode_num = 2
            for t in range(episode_num):
                # get one action
                if self.use_goal:
                    action = self.agent.get_action(state, goal, 0)
                else:
                    action = self.agent.get_action(state, 0)

                # step in the environment
                next_state, reward, done, dist, trans, _, _ = my_lab.step(action)

                # print the steps
                print(f"Current state = {state}, Action = {ACTION_LIST[action]}, Next state = {next_state}")

                # check terminal
                if done:
                    success_count += 1
                    break
                else:
                    state = next_state

        print("Success rate = {}".format(success_count/run_num))

    def navigate_with_local_policy(self):
        # init the environment
        self.update_map2d_and_maze3d(set_new_maze=True)
        # evaluation number
        run_num = 100
        # fail counter
        fail_count = 0
        success_count = 0
        # length counter
        length = 0
        # start testing experiment
        for r in range(run_num):
            # sample a start and goal position
            self.fix_start = False
            self.fix_goal = False
            # sample a pair of start and goal positions
            state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)

            # get the planned path
            path = [pos.tolist() + [0] for pos in self.env_map.path]
            # get sub-goals
            sub_goals_pos = []
            sub_goals_obs = []
            for i in range(0, len(path), 2):
                if i != 0:
                    sub_goals_pos.append(path[i])
                    if self.use_obs:
                        if not self.use_imagine:
                            goal_obs = self.env.get_random_observations(path[i])
                        else:
                            goal_loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded, path[i][0:2])
                            goal_obs = self.imagine_goal_obs(goal_loc_map)
                        sub_goals_obs.append(goal_obs)
            # check the final goal is in the list
            if not (path[-1] in sub_goals_pos):
                sub_goals_pos.append(path[-1])
                if use_obs:
                    sub_goals_obs.append(goal)
            length += len(path)
            if self.goal_dist == -1:
                distance = len(path)
            else:
                distance =self.goal_dist
            print("Run idx = {}, start pos = {}, goal pos = {}, dist = {}".format(r + 1, start_pos, goal_pos, distance))
            # # navigating between sub-goals
            # if not self.use_obs:
            #     nav_sub_goals = sub_goals_pos
            # else:
            #     nav_sub_goals = sub_goals_obs
            # maze_goal_pos = self.env.position_map2maze(goal_pos + [0], [self.maze_size, self.maze_size])
            # for t in range(100):
            #     action = self.agent.get_action(state, goal, 0)
            #     # action = random.sample(range(4), 1)[0]
            #     # step the environment and print info
            #     next_state, reward, done, dist, trans, _, _ = my_lab.step(action)
            #
            #     # update
            #     state = next_state
            #     # check termination
            #     if trans == maze_goal_pos:
            #         success_count += 1
            #         break
            # navigating between sub-goals
            if not self.use_obs:
                nav_sub_goals = sub_goals_pos
            else:
                nav_sub_goals = sub_goals_obs
            for idx, g in enumerate(nav_sub_goals):
                sub_goal_done = False
                max_time_step = 2
                maze_goal_pos = self.env.position_map2maze(sub_goals_pos[idx], [self.maze_size, self.maze_size])
                for t in range(max_time_step):
                    # get the action
                    action = self.agent.get_action(state, g, 0)
                    # action = random.sample(range(4), 1)[0]
                    # step the environment and print info
                    next_state, reward, done, dist, trans, _, _ = my_lab.step(action)

                    # update
                    state = next_state
                    # check termination
                    if self.use_obs:
                        tmp_sub_goal = maze_goal_pos
                    else:
                        tmp_sub_goal = g
                    if trans == tmp_sub_goal:
                        sub_goal_done = True
                        break
                if not sub_goal_done:
                    print("Fail to reach sub-goal {}".format(sub_goals_pos[idx]))
                    fail_count += 1
                    break

        print("Success rate = {}".format((run_num - fail_count)/run_num))
        print("Average len = {}".format(length / run_num))
        # print("Success rate = {}".format(success_count / run_num))
        # print("Average len = {}".format(length / run_num))

    def imagine_goal_obs(self, pos_loc_map):
        imagined_obs = []
        loc_map = torch.from_numpy(pos_loc_map).flatten().view(1, -1).float()
        for ori in self.orientations:
            z = torch.randn(1, 64)
            tmp_map = torch.cat(2 * [loc_map], dim=1)
            tmp_ori = torch.cat(2 * [ori.view(-1, 1 * 1 * 8).float()], dim=1)
            conditioned_z = torch.cat((z, tmp_map, tmp_ori), dim=1)
            obs_reconstructed, _ = self.cvae.decoder(conditioned_z)
            obs_reconstructed = obs_reconstructed.detach().squeeze(0).numpy().transpose(1, 2, 0) * 255
            imagined_obs.append(obs_reconstructed)
        return np.array(imagined_obs, dtype=np.uint8)

    def update_map2d_and_maze3d(self, set_new_maze=False):
        """
        Function is used to update the 2D map and the 3D maze.
        """
        # set maze configurations
        maze_configs = defaultdict(lambda: None)
        # set new maze flag
        if set_new_maze:
            # randomly select a maze
            self.maze_size = random.sample(self.maze_size_list, 1)[0]
            self.maze_seed = random.sample(self.maze_seed_list, 1)[0]
            print("Maze : {} - {}".format(self.maze_size, self.maze_seed))
            # initialize the map 2D
            self.env_map = mapper.RoughMap(self.maze_size, self.maze_seed, 3)
            init_map_pos = self.env_map.init_pos
            goal_map_pos = self.env_map.goal_pos
            # initialize the maze 3D
            maze_configs["maze_name"] = f"maze_{self.maze_size}x{self.maze_size}"  # string type name
            maze_configs["maze_size"] = [self.maze_size, self.maze_size]  # [int, int] list
            maze_configs["maze_seed"] = '1234'  # string type number
            maze_configs["maze_texture"] = random.sample(self.theme_list, 1)[0]  # string type name in theme_list
            maze_configs["maze_decal_freq"] = random.sample(self.decal_list, 1)[0]  # float number in decal_list
            maze_configs["maze_map_txt"] = "".join(self.env_map.map2d_txt)  # string type map
            maze_configs["maze_valid_pos"] = self.env_map.valid_pos  # list of valid positions
            # initialize the maze start and goal positions
            maze_configs["start_pos"] = self.env_map.init_pos + [0]  # start position on the txt map [rows, cols, orientation]
            maze_configs["goal_pos"] = self.env_map.goal_pos + [0]  # goal position on the txt map [rows, cols, orientation]
            # initialize the update flag
            maze_configs["update"] = True  # update flag
        else:
            init_pos, goal_pos = self.env_map.sample_random_start_goal_pos(self.fix_start, self.fix_goal, self.goal_dist)
            self.env_map.update_mapper(init_pos, goal_pos)
            # set the init and goal position
            init_map_pos = self.env_map.init_pos
            goal_map_pos = self.env_map.goal_pos
            # set the maze configurations
            maze_configs['start_pos'] = init_pos + [0]
            maze_configs['goal_pos'] = goal_pos + [0]
            maze_configs['maze_valid_pos'] = self.env_map.valid_pos
            maze_configs['update'] = False

        # obtain the state and goal observation
        state_obs, goal_obs, _, _ = self.env.reset(maze_configs)
        # return states and goals
        return state_obs, goal_obs, init_map_pos, goal_map_pos


if __name__ == '__main__':
    # random seed
    rnd_seed = 1234
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)
    # set parameters
    maze_size = 7
    run_local = True
    use_obs = True
    use_goal = True
    use_imagine = True
    goal_dist = 15
    seed = 0

    # set level name
    level_name = 'nav_random_maze'
    # necessary observations (correct: this is the egocentric observations (following the counter clock direction))
    observation_list = [
        'RGB.LOOK_PANORAMA_VIEW',
        'RGB.LOOK_TOP_DOWN_VIEW'
    ]
    observation_width = 32
    observation_height = 32
    observation_fps = 60
    # set configurations
    configurations = {
        'width': str(observation_width),
        'height': str(observation_height),
        'fps': str(observation_fps)
    }
    # create the environment
    my_lab = RandomMazeTileRaw(level_name,
                               observation_list,
                               configurations,
                               use_true_state=not use_obs,
                               reward_type="sparse-1",
                               dist_epsilon=1e-3)

    # load the agent
    if not use_obs:
        if not use_goal:
            my_agent = DQNAgent(use_true_state=True, use_small_obs=True)
            my_agent.policy_net.load_state_dict(
                torch.load(f"./results/5-9/ddqn_{maze_size}x{maze_size}_true_state_double_seed_{seed}.pt",
                           map_location=torch.device('cpu'))
            )
            print("Vanilla double DQN with true state.")
        else:
            my_agent = GoalDQNAgent(use_true_state=True, use_small_obs=True)
            my_agent.policy_net.load_state_dict(
                torch.load(f"./results/5-9/random_goal_ddqn_{maze_size}x{maze_size}_true_state_double_seed_{seed}.pt",
                           map_location=torch.device('cpu'))
            )
            print("Random Goal-conditioned double DQN with true state.")
    else:
        if not use_goal:
            my_agent = DQNAgent(use_true_state=False, use_small_obs=True)
            my_agent.policy_net.load_state_dict(
                torch.load(f"./results/5-9/ddqn_{maze_size}x{maze_size}_panorama_obs_double_seed_{seed}.pt",
                           map_location=torch.device('cpu'))
            )
            print("Vanilla double DQN with panorama observation.")
        else:
            my_agent = GoalDQNAgent(use_true_state=False, use_small_obs=True)
            my_agent.policy_net.load_state_dict(
                torch.load(f"./results/5-12/random_imagine_goal_ddqn_{maze_size}x{maze_size}_obs_double_random_maze_seed_{seed}.pt",
                           map_location=torch.device('cpu'))
            )
            print("Random Goal-conditioned double DQN with panorama HER observation.")
    my_agent.policy_net.eval()

    # run the agent
    myVis = VisualPolicy(env=my_lab,
                         agent=my_agent,
                         size=maze_size,
                         set_goal=use_goal,
                         fix_start=True,
                         fix_goal=True,
                         g_dist=goal_dist,
                         use_obs=use_obs,
                         use_imagine=use_imagine)
    if not run_local:
        myVis.run_fixed_start_goal_pos()
    else:
        myVis.navigate_with_local_policy()