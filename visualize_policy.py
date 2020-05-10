""" This script is a demo to control agent in a specific static maze environment in DeepMind Lab.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import random
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
    def __init__(self, env, agent, size, set_goal, fix_start, fix_goal, g_dist, use_obs):
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
        self.decal_list = [0.1]
        self.use_goal = set_goal
        self.use_obs = use_obs
        self.goal_dist = g_dist
        self.gamma = 0.99

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
        # episodes
        run_num = 100
        # fail counter
        fail_count = 0
        # start testing experiment
        for r in range(run_num):
            print("Run idx = {}".format(r+1))
            # sample a start and goal position
            self.fix_start = False
            self.fix_goal = False
            self.goal_dist = 2  # target distance
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
                        goal_obs = self.env.get_random_observations(path[i])
                        sub_goals_obs.append(goal_obs)
            # check the final goal is in the list
            if not (path[-1] in sub_goals_pos):
                sub_goals_pos.append(path[-1])
                if use_obs:
                    sub_goals_obs.append(goal)
            # print(path)
            print(start_pos, goal_pos)

            # navigating between sub-goals
            last_trans = self.env.position_map2maze(start_pos + [0], [self.maze_size, self.maze_size])
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
                    # step the environment and print info
                    next_state, reward, done, dist, trans, _, _ = my_lab.step(action)
                    # print("Current state = {}, Action = {}, Next state = {}, Goal = {}".format(last_trans, ACTION_LIST[action], trans, maze_goal_pos))
                    if t == 1 and idx == 0:
                        np.save(f'./{idx}_{t}_state.npy', state)
                        np.save(f'./{idx}_{t}_goal.npy', g)

                    # update
                    state = next_state
                    last_trans = trans
                    # check termination
                    if self.use_obs:
                        tmp_sub_goal = maze_goal_pos
                    else:
                        tmp_sub_goal = g
                    if trans == tmp_sub_goal:
                        sub_goal_done = True
                        # print("Reach goal = {}".format(sub_goals_pos[idx]))
                        break
                if not sub_goal_done:
                    print("Fail to reach sub-goal {}".format(sub_goals_pos[idx]))
                    fail_count += 1
                    break

        print("Success rate = {}".format((run_num - fail_count)/run_num))

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

    def test_navigate_with_local_policy(self):
        # init the environment
        self.update_map2d_and_maze3d(set_new_maze=True)
        # episodes
        run_num = 100
        # fail counter
        fail_count = 0
        # start testing experiment
        for r in range(run_num):
            print("Run idx = {}".format(r + 1))
            # sample a start and goal position
            self.fix_start = False
            self.fix_goal = False
            self.goal_dist = 12  # target distance
            # sample a pair of start and goal positions
            state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)

            # get the planned path
            path = [pos.tolist() + [0] for pos in self.env_map.path]
            # get sub-goals
            sub_goals_pos = []
            for i in range(0, len(path), 2):
                if i != 0:
                    sub_goals_pos.append(path[i])
            # check the final goal is in the list
            if not (path[-1] in sub_goals_pos):
                sub_goals_pos.append(path[-1])
            print(sub_goals_pos, start_pos, goal_pos)

            # navigating between sub-goals
            last_trans = self.env.position_map2maze(start_pos + [0], [self.maze_size, self.maze_size])
            start_pos = start_pos + [0]
            for g in sub_goals_pos:
                sub_goal_done = False
                max_time_step = 10
                maze_configs = defaultdict(lambda: None)
                init_pos = start_pos
                goal_pos = g
                self.env_map.update_mapper(init_pos[0:2], goal_pos[0:2])
                # set the maze configurations
                maze_configs['start_pos'] = init_pos
                maze_configs['goal_pos'] = goal_pos
                maze_configs['maze_valid_pos'] = self.env_map.valid_pos
                maze_configs['update'] = False
                # obtain the state and goal observation
                goal_position = self.env.position_map2maze(goal_pos, [self.maze_size, self.maze_size])
                state, goal, _, _ = self.env.reset(maze_configs)
                print(start_pos, goal_pos)
                for t in range(max_time_step):
                    # get the action
                    action = self.agent.get_action(state, goal, 0)
                    # step the environment and print info
                    next_state, reward, done, dist, trans, _, _ = my_lab.step(action)
                    # print("Current state = {}, Action = {}, Next state = {}, Goal = {}".format(last_trans,
                    #                                                                            ACTION_LIST[action],
                    #                                                                            trans, goal_position))
                    # update
                    state = next_state
                    last_trans = trans
                    # check termination
                    if trans == goal_position:
                        sub_goal_done = True
                        print("Reach goal = {}".format(g))
                        start_pos = g
                        break
                if not sub_goal_done:
                    print("Fail to reach sub-goal {}".format(g))
                    fail_count += 1
                    break

        print("Success rate = {}".format((run_num - fail_count) / run_num))


class TEST(object):
    def __init__(self):
        self.figs = None
        self.arrays = None
        self.img_artists = []

    def show_panorama_view_test(self, flag, state):
        observations = state
        print(state.shape)
        # init or update data
        if flag is None:
            self.fig, self.arrays = plt.subplots(3, 3)
            self.arrays[0, 1].set_title("Front view")
            self.arrays[0, 1].axis("off")
            self.img_artists.append(self.arrays[0, 1].imshow(observations[0]))
            self.arrays[0, 0].set_title("Front-left view")
            self.arrays[0, 0].axis("off")
            self.img_artists.append(self.arrays[0, 0].imshow(observations[1]))
            self.arrays[1, 0].set_title("Left view")
            self.arrays[1, 0].axis("off")
            self.img_artists.append(self.arrays[1, 0].imshow(observations[2]))
            self.arrays[1, 1].set_title("Top-down view")
            self.arrays[1, 1].axis("off")
            # self.img_artists.append(self.arrays[1, 1].imshow(ndimage.rotate(self._top_down_obs, -90)))
            self.arrays[2, 0].set_title("Back-left view")
            self.arrays[2, 0].axis("off")
            self.img_artists.append(self.arrays[2, 0].imshow(observations[3]))
            self.arrays[2, 1].set_title("Back view")
            self.arrays[2, 1].axis("off")
            self.img_artists.append(self.arrays[2, 1].imshow(observations[4]))
            self.arrays[2, 2].set_title("Back-right view")
            self.arrays[2, 2].axis("off")
            self.img_artists.append(self.arrays[2, 2].imshow(observations[5]))
            self.arrays[1, 2].set_title("Right view")
            self.arrays[1, 2].axis("off")
            self.img_artists.append(self.arrays[1, 2].imshow(observations[6]))
            self.arrays[0, 2].set_title("Front-right view")
            self.arrays[0, 2].axis("off")
            self.img_artists.append(self.arrays[0, 2].imshow(observations[7]))
        else:
            self.img_artists[0].set_data(observations[0])
            self.img_artists[1].set_data(observations[1])
            self.img_artists[2].set_data(observations[2])
            # self.img_artists[3].set_data(ndimage.rotate(self._top_down_obs, -90))
            self.img_artists[3].set_data(observations[3])
            self.img_artists[4].set_data(observations[4])
            self.img_artists[5].set_data(observations[5])
            self.img_artists[6].set_data(observations[6])
            self.img_artists[7].set_data(observations[7])
        self.fig.canvas.draw()
        plt.pause(0.0001)
        return self.fig

    def test_demo(self):
        state_1 = np.load('./1_1_state.npy')
        goal_1 = np.load('./1_1_goal.npy')

        state_2 = np.load('./0_1_state.npy')
        goal_2 = np.load('./0_1_goal.npy')

        q_values_1 = my_agent.get_action(state_2, goal_1, 0)
        q_values_2 = my_agent.get_action(state_1, goal_2, 0)

        # fig1 = myTest.show_panorama_view_test(None, goal_1)
        # fig2 = myTest.show_panorama_view_test(None, goal_2)
        # fig2 = myTest.show_panorama_view_test(None, abs(state_2 - state_1))
        # plt.show()
        print('Hello world')


if __name__ == '__main__':
    random.seed(1234)
    # set parameters
    maze_size = 9
    run_local = True
    use_obs = True
    use_goal = True
    goal_dist = 2
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
                torch.load(f"./results/5-4/ddqn_{maze_size}x{maze_size}_true_state_double_seed_{seed}.pt",
                           map_location=torch.device('cpu'))
            )
        else:
            my_agent = GoalDQNAgent(use_true_state=True, use_small_obs=True)
            my_agent.policy_net.load_state_dict(
                torch.load(f"./results/5-4/random_goal_ddqn_{maze_size}x{maze_size}_true_state_double_seed_{seed}.pt",
                           map_location=torch.device('cpu'))
            )
    else:
        if not use_goal:
            my_agent = DQNAgent(use_true_state=False, use_small_obs=True)
            my_agent.policy_net.load_state_dict(
                torch.load(f"./results/5-4/ddqn_{maze_size}x{maze_size}_panorama_obs_double_seed_{seed}.pt",
                           map_location=torch.device('cpu'))
            )
        else:
            my_agent = GoalDQNAgent(use_true_state=False, use_small_obs=True)
            my_agent.policy_net.load_state_dict(
                torch.load(f"./results/5-4/random_goal_ddqn_{maze_size}x{maze_size}_obs_double_seed_{seed}.pt",
                           map_location=torch.device('cpu'))
            )
    my_agent.policy_net.eval()

    # run the agent
    myVis = VisualPolicy(env=my_lab,
                         agent=my_agent,
                         size=maze_size,
                         set_goal=use_goal,
                         fix_start=True,
                         fix_goal=True,
                         g_dist=goal_dist,
                         use_obs=use_obs)
    if not run_local:
        myVis.run_fixed_start_goal_pos()
    else:
        # myVis.run_random_start_goal_pos()
        myVis.test_navigate_with_local_policy()