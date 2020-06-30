""" This script is a demo to control agent in a specific static maze environment in DeepMind Lab.
        - Evaluate baselines:
            - Random policy
            - Double DQN + HER
        - Evaluate my method

        Usage: The user will indicate the mazes (size and seeds) to be evaluated. Both the trained and test mazes
               should be evaluated.

               This evaluation is for the second tile version.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import random
from model import VAE
from test.GoalDQNAgent import GoalDQNAgent
from utils import mapper
from collections import defaultdict
from envs.LabEnvV2 import RandomMazeTileRaw
import numpy as np
import pickle
from utils import searchAlg
import IPython.terminal.debugger as Debug
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ACTION_LIST = ['up', 'down', 'left', 'right']


class EvalPolicy(object):
    def __init__(self, env, agent, size_list, seed_list, dist_list, res_path, res_name, args):
        # evaluation env object
        self.env = env
        # 2D rough map object
        self.env_map = None
        # agent object
        self.agent = agent
        # mazes to be evaluated
        self.maze_size_list = size_list
        self.maze_seed_list = seed_list
        # evaluation protocol
        self.fix_start = False
        self.fix_goal = False
        self.theme_list = ['MISHMASH']
        self.decal_list = [0.001]
        self.use_goal = None
        self.use_obs = not args.use_true_state
        self.use_imagine = args.use_imagine
        self.goal_dist = dist_list
        self.gamma = 0.99
        self.run_num = args.run_num
        # parameters for generating fake observations
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
        # self.cvae.load_state_dict(torch.load("/mnt/cheng_results/trained_model/VAE/small_obs_L64_B8.pt", map_location=torch.device('cuda:0')))
        self.cvae.load_state_dict(torch.load("./results/vae/model/small_obs_L64_B8.pt", map_location=torch.device('cpu')))
        self.cvae.eval()

        # save parameters
        self.save_path = res_path
        self.file_name = res_name

    def eval_random_policy(self):
        # save the data
        eval_results = defaultdict()
        # loop all the mazes
        for m_size in self.maze_size_list:
            for m_seed in self.maze_seed_list:
                # loop all distances
                for g_dist in self.goal_dist:
                    # init the 3D environment
                    self.update_map2d_and_maze3d(set_new_maze=True, maze_size=m_size, maze_seed=m_seed, dist=g_dist)

                    # store evaluation results
                    run_num = self.run_num
                    success_count = 0
                    run_count = 0
                    # start testing
                    for r in range(run_num):
                        # sample a random pair of start and goal positions
                        self.fix_start = False
                        self.fix_goal = False
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False,
                                                                                        maze_size=m_size,
                                                                                        maze_seed=m_seed,
                                                                                        dist=g_dist)
                        # check the sampled distance
                        if len(self.env_map.path) < g_dist + 1:
                            print("Sampled pair unsatisfied.")
                            continue

                        # print the current evaluation
                        run_count += 1
                        print("{}-{}: Run idx = {}, Init = {}, Goal = {}, Dist = {}".format(m_size,
                                                                                            m_seed,
                                                                                            r,
                                                                                            start_pos,
                                                                                            goal_pos,
                                                                                            len(self.env_map.path)))
                        # set the maximal steps = 3 * optimal steps
                        max_episode_len = (len(self.env_map.path) - 1) * 3
                        for t in range(max_episode_len):
                            # randomly sample an action
                            action = random.sample(range(4), 1)[0]
                            # step in the environment
                            next_state, reward, done, dist, trans, _, _ = my_lab.step(action)

                            # check terminal
                            if done:
                                success_count += 1
                                break

                    # avoid 0 denominator
                    if run_count > 0:
                        # print the results
                        print("Success rate = {}".format(success_count / run_count))
                        # store the results
                        eval_results[f"{m_size}-{m_seed}-{g_dist}"] = success_count / run_count
                    else:
                        eval_results[f"{m_size}-{m_seed}-{g_dist}"] = None
        # print info
        print("Evaluation finished")
        # save the dictionary as txt file
        save_name = '/'.join([self.save_path, f"eval_{self.file_name}_policy.txt"])
        with open(save_name, 'w') as f:
            for key, val in eval_results.items():
                tmp_str = key + ' ' + str(val) + '\n'
                f.write(tmp_str)
            f.close()

    def eval_her_policy(self):
        # save the data
        eval_results = defaultdict()
        # loop all the mazes
        for m_size in self.maze_size_list:
            for m_seed in self.maze_seed_list:
                # loop all distances
                for g_dist in self.goal_dist:
                    # init the 3D environment
                    self.update_map2d_and_maze3d(set_new_maze=True, maze_size=m_size, maze_seed=m_seed, dist=g_dist)

                    # store evaluation results
                    run_num = self.run_num
                    success_count = 0
                    run_count = 0
                    # start testing
                    for r in range(run_num):
                        # sample a random pair of start and goal positions
                        self.fix_start = False
                        self.fix_goal = False
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False,
                                                                                        maze_size=m_size,
                                                                                        maze_seed=m_seed,
                                                                                        dist=g_dist)
                        if len(self.env_map.path) < g_dist + 1:
                            print("Sampled pair unsatisfied. Skipped")
                            continue
                        run_count += 1
                        # set the maximal steps
                        print("{}-{}: Run idx = {}, Init = {}, Goal = {}, Dist = {}".format(m_size,
                                                                                            m_seed,
                                                                                            r,
                                                                                            start_pos,
                                                                                            goal_pos,
                                                                                            len(self.env_map.path)))
                        max_episode_len = (len(self.env_map.path) - 1)
                        for t in range(max_episode_len):
                            # randomly sample an action
                            action = self.agent.get_action(state, goal, 0)
                            # step in the environment
                            next_state, reward, done, dist, trans, _, _ = my_lab.step(action)
                            state = next_state
                            # check terminal
                            if done:
                                success_count += 1
                                break

                    if run_count > 0:
                        # print the results
                        print("Success rate = {}".format(success_count / run_count))
                        # store the results
                        eval_results[f"{m_size}-{m_seed}-{g_dist}"] = success_count / run_count
                    else:
                        eval_results[f"{m_size}-{m_seed}-{g_dist}"] = None
        # print info
        print("Evaluation finished")
        # save the dictionary as txt file
        save_name = '/'.join([self.save_path, f"eval_{self.file_name}_policy.txt"])
        with open(save_name, 'w') as f:
            for key, val in eval_results.items():
                tmp_str = key + ' ' + str(val) + '\n'
                f.write(tmp_str)
            f.close()

    def eval_navigate_with_local_policy(self):
        # store training data
        eval_results = defaultdict()
        # loop all the mazes
        for m_size in self.maze_size_list:
            for m_seed in self.maze_seed_list:
                # loop all the distance
                for g_dist in self.goal_dist:
                    # sample a 3D maze environment
                    self.update_map2d_and_maze3d(set_new_maze=True, maze_size=m_size, maze_seed=m_seed, dist=g_dist)
                    # evaluation number
                    run_num = self.run_num
                    run_count = 0
                    # fail counter
                    fail_count = 0
                    # store the actions
                    for r in range(run_num):
                        act_list = []
                        # sample a start and goal position
                        self.fix_start = False
                        self.fix_goal = False
                        # sample a pair of start and goal positions
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False,
                                                                                        maze_size=m_size,
                                                                                        maze_seed=m_seed,
                                                                                        dist=g_dist)
                        # get the planned path
                        path = [pos.tolist() + [0] for pos in self.env_map.path]
                        # get sub-goals
                        sub_goals_pos = []
                        sub_goals_obs = []
                        for i in range(0, len(path), 1):
                            if i != 0:
                                # save the sub goal position
                                sub_goals_pos.append(path[i])
                                # save the sub goal observation if use observation
                                if self.use_obs:
                                    # save the true observation if not use imagination
                                    if not self.use_imagine:
                                        goal_obs = self.env.get_random_observations_tile(path[i])
                                    else:  # save imagined observation if use imagination
                                        goal_loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded, path[i][0:2])
                                        goal_obs = self.imagine_goal_obs(goal_loc_map)
                                    # save the observation
                                    sub_goals_obs.append(goal_obs)

                        # check the final goal is in the list
                        if not (path[-1] in sub_goals_pos):
                            # add the goal position
                            sub_goals_pos.append(path[-1])
                            # add the goal observation (True observation)
                            if not args.use_true_state:
                                sub_goals_obs.append(goal)

                        # check the distance validation
                        if g_dist == -1:
                            distance = len(path)
                        else:
                            distance = len(path)
                        if distance < g_dist + 1:
                            print(distance, g_dist + 1)
                            print(f"Run idx = {r + 1} not satisfied.")
                            continue
                        else:
                            run_count += 1

                        # print info for validation sampled start-goal position
                        print("{}-{}: Run idx = {}, start pos = {}, goal pos = {}, dist = {}".format(m_size, m_seed, r + 1, start_pos, goal_pos, distance))

                        # navigating between sub-goals
                        if not self.use_obs:
                            nav_sub_goals = sub_goals_pos
                        else:
                            nav_sub_goals = sub_goals_obs
                        for idx, g in enumerate(nav_sub_goals):
                            # flag for sub-goal navigation
                            sub_goal_done = False
                            # maximal steps for sub-goal navigation
                            max_time_step = 2
                            # convert the goal position to maze position
                            maze_goal_pos = self.env.position_map2maze(sub_goals_pos[idx], [m_size, m_size])

                            for t in range(max_time_step):
                                # get the action
                                action = self.agent.get_action(state, g, 0)
                                # save the action
                                act_list.append(ACTION_LIST[action])
                                # step the environment and print info
                                next_state, reward, done, dist, next_trans, _, _ = my_lab.step(action)
                                # update

                                state = next_state
                                # check termination
                                if self.use_obs:
                                    tmp_sub_goal = maze_goal_pos
                                else:
                                    tmp_sub_goal = g
                                if abs(np.sum(next_trans - np.array(tmp_sub_goal))) < 1:
                                    sub_goal_done = True
                                    break
                            if not sub_goal_done:
                                print("Fail to reach sub-goal {}".format(sub_goals_pos[idx]))
                                print(f"Failed actions = {act_list}")
                                fail_count += 1
                                break
                        if done:
                            print("Success navigation action list: ", act_list)
                            print("Ground truth: ", self.env_map.map_act)
                            print("------------------------------------------------------------------")

                    if run_count > 0:
                        print("Success rate = {}".format((run_count - fail_count) / run_count))
                        eval_results[f"{m_size}-{m_seed}-{g_dist}"] = (run_count - fail_count) / run_count
                    else:
                        eval_results[f"{m_size}-{m_seed}-{g_dist}"] = None

        # print info
        print("Evaluation finished")
        # save the dictionary as txt file
        save_name = '/'.join([self.save_path, f"eval_{self.file_name}_policy.txt"])
        with open(save_name, 'w') as f:
            for key, val in eval_results.items():
                tmp_str = key + ' ' + str(val) + '\n'
                f.write(tmp_str)
            f.close()

    def eval_navigate_with_local_policy_loop_entire(self):
        # store training data
        eval_results = defaultdict()
        # loop all the mazes
        for m_size in self.maze_size_list:
            mean_success = []
            for m_seed in self.maze_seed_list:
                # loop all the distance
                self.update_map2d_and_maze3d(set_new_maze=True, maze_size=m_size, maze_seed=m_seed, dist=-1)
                total_pairs_dict = self.load_pair_data(m_size, m_seed)
                pairs_dict = {'start': None, 'goal': None}
                for g_dist in self.goal_dist:
                    # obtain all the pairs
                    pairs_dict['start'] = total_pairs_dict[str(g_dist)][0]
                    pairs_dict['goal'] = total_pairs_dict[str(g_dist)][1]
                    # loop all possible pairs
                    fail_count = 0
                    run_count = len(pairs_dict['start']) * 1
                    count = 1
                    for s_pos, g_pos in zip(pairs_dict['start'], pairs_dict['goal']):
                        # s_pos = [13, 4]
                        # g_pos = [13, 3]
                        # forward test
                        act_list = []
                        state, goal, start_pos, goal_pos = self.update_maze_from_pos(s_pos, g_pos)
                        # get the planned path
                        path = [pos.tolist() + [0] for pos in self.env_map.path]
                        # get sub-goals
                        sub_goals_pos = []
                        sub_goals_obs = []
                        for i in range(0, len(path), 1):
                            if i != 0:
                                # save the sub goal position
                                sub_goals_pos.append(path[i])
                                # save the sub goal observation if use observation
                                if self.use_obs:
                                    # save the true observation if not use imagination
                                    if not self.use_imagine:
                                        goal_obs = self.env.get_random_observations_tile(path[i])
                                    else:  # save imagined observation if use imagination
                                        goal_loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded,
                                                                            path[i][0:2])
                                        # print('imagine')
                                        goal_obs = self.imagine_goal_obs(goal_loc_map)
                                    # save the observation
                                    sub_goals_obs.append(goal_obs)

                        # check the final goal is in the list
                        if not (path[-1] in sub_goals_pos):
                            # add the goal position
                            sub_goals_pos.append(path[-1])
                            # add the goal observation (True observation)
                            if not args.use_true_state:
                                sub_goals_obs.append(goal) 

                        # navigating between sub-goals
                        if not self.use_obs:
                            nav_sub_goals = sub_goals_pos
                        else:
                            nav_sub_goals = sub_goals_obs

                        current_trans = self.env.position_map2maze(path[0], [m_size, m_size])
                        for idx, g in enumerate(nav_sub_goals):
                            # flag for sub-goal navigation
                            sub_goal_done = False
                            # maximal steps for sub-goal navigation
                            max_time_step = 1
                            # convert the goal position to maze position
                            maze_goal_pos = self.env.position_map2maze(sub_goals_pos[idx], [m_size, m_size])
                            # state_loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded, self.env.position_maze2map(current_trans, [m_size, m_size])[0:2])
                            # state = self.imagine_goal_obs(state_loc_map)
                            for t in range(max_time_step):
                                # get the action
                                with torch.no_grad():
                                    action = self.agent.get_action(state, g, 0)
                                # save the action
                                act_list.append(ACTION_LIST[action])
                                # step the environment and print info
                                next_state, reward, done, dist, next_trans, _, _ = my_lab.step(action)
                                # check termination
                                if self.use_obs:
                                    tmp_sub_goal = maze_goal_pos
                                else:
                                    tmp_sub_goal = g

                                # update
                                state = next_state
                                current_trans = next_trans

                                if np.sum(abs(next_trans - np.array(tmp_sub_goal))) == 0:
                                    sub_goal_done = True
                                    break
                            if not sub_goal_done:
                                print("Fail to reach sub-goal {}".format(sub_goals_pos[idx]))
                                print(f"Failed actions = {act_list}")
                                fail_count += 1
                                break
                                # print info for validation sampled start-goal position
                        print("{}-{}: Start pos = {}, Goal pos = {}, Dist = {}, Done = {}, Acts = {}".format(
                                    m_size,
                                    m_seed,
                                    start_pos,
                                    goal_pos,
                                    g_dist,
                                    done,
                                    act_list))
                        
                        # reverse the start and goal position
                        act_list = []
                        tmp = s_pos
                        s_pos = g_pos
                        g_pos = tmp
                        state, goal, start_pos, goal_pos = self.update_maze_from_pos(s_pos, g_pos)
                        # get the planned path
                        path = [pos.tolist() + [0] for pos in self.env_map.path]
                        # get sub-goals
                        sub_goals_pos = []
                        sub_goals_obs = []
                        for i in range(0, len(path), 1):
                            if i != 0:
                                # save the sub goal position
                                sub_goals_pos.append(path[i])
                                # save the sub goal observation if use observation
                                if self.use_obs:
                                    # save the true observation if not use imagination
                                    if not self.use_imagine:
                                        goal_obs = self.env.get_random_observations_tile(path[i])
                                    else:  # save imagined observation if use imagination
                                        goal_loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded,
                                                                            path[i][0:2])
                                        goal_obs = self.imagine_goal_obs(goal_loc_map)
                                    # save the observation
                                    sub_goals_obs.append(goal_obs)

                        # check the final goal is in the list
                        if not (path[-1] in sub_goals_pos):
                            # add the goal position
                            sub_goals_pos.append(path[-1])
                            # add the goal observation (True observation)
                            if not args.use_true_state:
                                sub_goals_obs.append(goal)

                        # navigating between sub-goals
                        if not self.use_obs:
                            nav_sub_goals = sub_goals_pos
                        else:
                            nav_sub_goals = sub_goals_obs

                        current_trans = self.env.position_map2maze(path[0], [m_size, m_size])
                        for idx, g in enumerate(nav_sub_goals):
                            # flag for sub-goal navigation
                            sub_goal_done = False
                            # maximal steps for sub-goal navigation
                            max_time_step = 1
                            # convert the goal position to maze position
                            maze_goal_pos = self.env.position_map2maze(sub_goals_pos[idx], [m_size, m_size])
                            # state_loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded, self.env.position_maze2map(current_trans, [m_size, m_size])[0:2])
                            # state = self.imagine_goal_obs(state_loc_map)
                            for t in range(max_time_step):
                                # get the action
                                action = self.agent.get_action(state, g, 0)
                                # save the action
                                act_list.append(ACTION_LIST[action])
                                # step the environment and print info
                                next_state, reward, done, dist, next_trans, _, _ = my_lab.step(action)
                                # print(f"Current pos = {current_trans}, action = {ACTION_LIST[action]}, next_state = {next_trans}, done = {done}, goal = {maze_goal_pos}")
                                # update
                                state = next_state
                                current_trans = next_trans
                                # check termination
                                if self.use_obs:
                                    tmp_sub_goal = maze_goal_pos
                                else:
                                    tmp_sub_goal = g
                                if np.sum(abs(next_trans - np.array(tmp_sub_goal))) == 0:
                                    sub_goal_done = True
                                    break
                            if not sub_goal_done:
                                print("Fail to reach sub-goal {}".format(sub_goals_pos[idx]))
                                print(f"Failed actions = {act_list}")
                                fail_count += 1
                                break
                        # print info for validation sampled start-goal position
                        print("{}-{}: Start pos = {}, Goal pos = {}, Dist = {}, Done = {}, Acts = {}".format(m_size,
                                                                                                             m_seed,
                                                                                                             start_pos,
                                                                                                             goal_pos,
                                                                                                             g_dist,
                                                                                                             done,
                                                                                                             act_list))
                        print('------------------------------------------------------------------------------------')
                    
                    print("Success rate = {}".format((run_count - fail_count) / run_count))
                    mean_success.append((run_count - fail_count) / run_count)
                    eval_results[f"{m_size}-{m_seed}-{g_dist}"] = (run_count - fail_count) / run_count
            print(np.mean(mean_success))
        # print info
        print("Evaluation finished")
        # save the dictionary as txt file
        save_name = '/'.join([self.save_path, f"eval_{self.file_name}_policy.txt"])
        with open(save_name, 'w') as f:
            for key, val in eval_results.items():
                tmp_str = key + ' ' + str(val) + '\n'
                f.write(tmp_str)
            f.close()

    def eval_navigate_with_hybrid_local_policy_using_dynamic_behavior_map(self):
        # store training data
        eval_results = defaultdict()
        # loop all the mazes
        for m_size in self.maze_size_list:
            for m_seed in self.maze_seed_list:
                # initialize lab the environment
                print(f"Init maze {m_size}-{m_seed} environment")
                self.update_map2d_and_maze3d(set_new_maze=True, maze_size=m_size, maze_seed=m_seed, dist=-1)
                # build the initial dynamic behavior map
                print(f"Init the dynamic behavior map")
                mlb_map = self.build_mlb_from_2d_map(max_edge_len=1)
                mlb_graph = csr_matrix(mlb_map)
                # obtain all the distance pairs
                print(f"Load all pairs")
                total_pairs_dict = self.load_pair_data(m_size, m_seed)
                pairs_dict = {'start': None, 'goal': None}
                # loop all possible distance
                for g_dist in self.goal_dist:
                    # get pairs with specific distance
                    pairs_dict['start'] = total_pairs_dict[str(g_dist)][0]
                    pairs_dict['goal'] = total_pairs_dict[str(g_dist)][1]
                    total_sub_pair_num = len(pairs_dict['start'])
                    # evaluate counter
                    success_counter = 0
                    total_runs = (total_sub_pair_num * 2) if (total_sub_pair_num * 2) <= 100 else 100
                    # loop all possible pairs
                    print(f"maze: {m_size}-{m_seed}-{g_dist}")
                    # for s_pos, g_pos in zip(pairs_dict['start'], pairs_dict['goal']):
                    for run in range(total_runs):
                        # sample a start-goal pair
                        pair_idx = random.sample(range(total_sub_pair_num), 1)[0]
                        # with probability 0.5, reverse the order
                        if random.uniform(0, 1) < 0.5:
                            s_pos = pairs_dict['start'][pair_idx]
                            g_pos = pairs_dict['goal'][pair_idx]
                        else:
                            s_pos = pairs_dict['goal'][pair_idx]
                            g_pos = pairs_dict['start'][pair_idx]
                        # set the environment based on the sampled start and goal positions
                        state_obs, goal_obs, start_pos, goal_pos = self.update_maze_from_pos(start_pos=s_pos,
                                                                                             goal_pos=g_pos)
                        # set the budget for the navigation
                        max_time_steps = 100
                        # record the position on the map to terminate the sub-goal navigation
                        # currently, the signal is return from the environment
                        # sub-goal termination?
                        state_pos = start_pos
                        # plan for every time steps
                        t_counter = 0
                        nav_done = False
                        fail_count = 0
                        while t_counter < max_time_steps:
                            # get the next way point: observation, position (map)
                            next_obs, next_pos_map = self.mlb_search_next_waypoint(state_pos, goal_pos, mlb_graph)
                            # check whether there is a connected path
                            if next_obs is None or next_pos_map is None:
                                print(f"None path is found between {state_pos} and {goal_pos}")
                                break
                            # compute the pos in the environment to decide the sub-goal termination
                            next_pos_maze = self.env.position_map2maze(next_pos_map + [0], [m_size, m_size])
                            # way point navigation budget
                            max_steps_per_goal = 10
                            sub_nav_done = False
                            sub_action_list = []
                            while max_steps_per_goal > 0:
                                # get the action
                                action = self.agent.get_action(state_obs, next_obs, 0)
                                sub_action_list.append(action)
                                # take the action
                                next_state, reward, done, dist, next_trans, _, _ = my_lab.step(action)
                                # update the
                                nav_done = done
                                state_obs = next_state
                                # increase and decrease the indicator
                                t_counter += 1
                                max_steps_per_goal -= 1
                                # check terminal
                                if np.sum(abs(next_trans - np.array(next_pos_maze))) == 0:
                                    # print(f'Reach the way point {next_pos_map} from {state_pos} with action = {ACTION_LIST[action]}')
                                    sub_nav_done = True
                                    break
                            # if agent fail to reach the way point
                            if not sub_nav_done:
                                print(f"Fail to reach the way point {next_pos_map}")
                                fail_count += 1
                                # agent return to previous position
                                # Is there a better way to return to the old position?
                                reversed_actions = [1, 0, 3, 2]
                                for action in reversed(sub_action_list):
                                    reversed_act = reversed_actions[action]
                                    next_state, reward, _, dist, next_trans, _, _ = my_lab.step(reversed_act)
                                    state_obs = next_state
                                    t_counter += 1
                                # update the graph
                                state_idx = self.env_map.valid_pos.index(state_pos)
                                goal_idx = self.env_map.valid_pos.index(next_pos_map)
                                # based on the observation variance, I will let the agent try several times
                                if fail_count > 3:
                                    mlb_map[state_idx, goal_idx] = 0
                                    fail_count = 0
                                    mlb_graph = csr_matrix(mlb_map)
                                # if it fails for twice, I will use the imagination to guess an action
                                if fail_count == 2:
                                    loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded, state_pos)
                                    state_obs_imagined = self.imagine_goal_obs(loc_map)
                                    state_obs = state_obs_imagined
                            else:
                                state_pos = next_pos_map

                            # check the final terminal
                            if nav_done:
                                success_counter += 1
                                break
                        print(f"Run {run}: Start = {start_pos}, Goal = {goal_pos}, Dist = {len(self.env_map.path)}, Done = {nav_done}")
                        print(f"--------------------------------------------------------------------")
                    # show results
                    print(f"Mean successful rate for distance = {g_dist} is {np.round(success_counter / total_runs, 2)}")
                    eval_results[f"{m_size}-{m_seed}-{g_dist}"] = np.round(success_counter / total_runs, 2)
                # print info
                print("Evaluation finished")
                # save the dictionary as txt file
                save_name = '/'.join([self.save_path, f"eval_{self.file_name}_policy.txt"])
                with open(save_name, 'w') as f:
                    for key, val in eval_results.items():
                        tmp_str = key + ' ' + str(val) + '\n'
                        f.write(tmp_str)
                    f.close()

    # search for the action using mlb
    def mlb_search_next_waypoint(self, state_pos, goal_pos, mlb_graph):
        # get next way point
        next_waypoint_pos = self.mlb_get_next_waypoint(state_pos, goal_pos, mlb_graph)
        if next_waypoint_pos != -9999:
            # generate next way point observation
            waypoint_loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded, next_waypoint_pos)
            next_waypoint_obs = self.imagine_goal_obs(waypoint_loc_map)
            return next_waypoint_obs, next_waypoint_pos
        else:
            return None, None

    def mlb_get_next_waypoint(self, state_pos, goal_pos, graph):
        # get the position indices
        state_idx = self.env_map.valid_pos.index(state_pos)
        goal_idx = self.env_map.valid_pos.index(goal_pos)
        # search for the shortest path using Dijkstra's algorithm
        dist_matrix, predecessors = dijkstra(csgraph=graph, directed=True, indices=state_idx, return_predecessors=True)
        # generate the path
        path = []
        idx = goal_idx
        while idx != state_idx:
            path.append(self.env_map.valid_pos[idx])
            idx = predecessors[idx]
            if idx == -9999:
                break

        # deal with the case when there is no path
        if idx != -9999:
            next_waypoint_pos = path[-1]
            path.append(self.env_map.valid_pos[state_idx])
            return next_waypoint_pos
        else:
            return -9999

    def build_mlb_from_2d_map(self, max_edge_len=1):
        # get all the valid positions on the map
        map_valid_pos = self.env_map.valid_pos
        # get the position number
        map_valid_pos_num = len(map_valid_pos)
        # maximal edge length
        max_dist = max_edge_len + 1
        # construct the graph
        mlb_sparse_matrix = np.zeros((map_valid_pos_num, map_valid_pos_num))
        # start the evaluation
        for i in range(map_valid_pos_num):
            for j in range(map_valid_pos_num):
                if i == j:
                    continue
                # get the start position and goal position
                s_pos = map_valid_pos[i]
                g_pos = map_valid_pos[j]
                # get the path and action on the map
                tmp_path = searchAlg.A_star(self.env_map.map2d_grid, s_pos, g_pos)
                # we only consider few steps away
                if len(tmp_path) > max_dist:
                    continue
                # assume fully connectivity
                print(f"Start = {s_pos}, Goal = {g_pos}, Distance = {len(tmp_path)}")
                mlb_sparse_matrix[i, j] = len(tmp_path) - 1
        return mlb_sparse_matrix

    def load_pair_data(self, m_size, m_seed):
        # path = f'/mnt/cheng_results/map/maze_{m_size}_{m_seed}.pkl'
        path = f'./results/6-16/maze_{m_size}_{m_seed}.pkl'
        f = open(path, 'rb')
        return pickle.load(f)

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

    def update_map2d_and_maze3d(self,
                                set_new_maze=False,
                                maze_size=5,
                                maze_seed=0,
                                dist=-1):
        """
        Function is used to update the 2D map and the 3D maze.
        """
        # set maze configurations
        maze_configs = defaultdict(lambda: None)
        # set new maze flag
        if set_new_maze:
            # initialize the map 2D
            self.env_map = mapper.RoughMap(maze_size, maze_seed, 3)
            init_map_pos = self.env_map.init_pos
            goal_map_pos = self.env_map.goal_pos
            # initialize the maze 3D
            maze_configs["maze_name"] = f"maze_{maze_size}_{maze_seed}"  # string type name
            maze_configs["maze_size"] = [maze_size, maze_size]  # [int, int] list
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
            # sample a new start-goal pair
            init_pos, goal_pos = self.env_map.sample_random_start_goal_pos(self.fix_start, self.fix_goal, dist)
            # # set the init and goal position
            init_map_pos = self.env_map.init_pos
            goal_map_pos = self.env_map.goal_pos
            self.env_map.update_mapper(init_map_pos, goal_map_pos)
            # set the maze configurations
            maze_configs['start_pos'] = init_pos + [0]
            maze_configs['goal_pos'] = goal_pos + [0]
            maze_configs['maze_valid_pos'] = self.env_map.valid_pos
            maze_configs["maze_seed"] = '1234'  # string type number
            maze_configs['update'] = False

        # obtain the state and goal observation
        state_obs, goal_obs, _, _ = self.env.reset(maze_configs)
        # return states and goals
        return state_obs, goal_obs, init_map_pos, goal_map_pos

    def update_maze_from_pos(self, start_pos, goal_pos): 
        maze_configs = defaultdict(lambda: None)
        #print(f"Set start = {start_pos}, goal = {goal_pos}")
        self.env_map.update_mapper(start_pos, goal_pos)
        # set the maze configurations
        maze_configs['start_pos'] = self.env_map.init_pos + [0]
        maze_configs['goal_pos'] = self.env_map.goal_pos + [0]
        maze_configs['maze_valid_pos'] = self.env_map.valid_pos
        maze_configs["maze_seed"] = '1234'  # string type number
        maze_configs['update'] = False
        # obtain the state and goal observation
        state_obs, goal_obs, _, _ = self.env.reset(maze_configs)
        # return states and goals
        return state_obs, goal_obs, start_pos, goal_pos


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_mode", type=str, default='random-policy', help="Evaluation mode: random-policy, "
                                                                               "her-policy, or imagine-local-policy")
    parser.add_argument("--maze_size_list", type=str, default='5', help="Maze size list")
    parser.add_argument("--maze_seed_list", type=str, default='0', help="Maze seed list")
    parser.add_argument("--distance_list", type=str, default="1", help="Distance list")
    parser.add_argument("--save_path", type=str, default='./', help="Save path")
    parser.add_argument("--file_name", type=str, default="random", help="File name")
    parser.add_argument("--run_num", type=int, default=100, help="Number of evaluation run")
    parser.add_argument("--use_true_state", type=str, default="True", help="Whether use the true state")
    parser.add_argument("--use_goal", type=str, default="True", help="Whether use the goal conditioned policy")
    parser.add_argument("--use_imagine", type=str, default="True", help="Whether use the imagined observation")
    parser.add_argument("--use_rescale", type=str, default="True", help="Whether use the rescaled observation")
    return parser.parse_args()


# convert the True/False from str to bool
def strTobool(inputs):
    inputs.use_true_state = True if inputs.use_true_state == "True" else False
    inputs.use_goal = True if inputs.use_goal == "True" else False
    inputs.use_imagine = True if inputs.use_imagine == "True" else False
    inputs.use_rescale = True if inputs.use_rescale == "True" else False
    return inputs


if __name__ == '__main__':
    # random seed
    rnd_seed = 1234
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)

    # set the evaluation model
    args = parse_input()
    args = strTobool(args)
    eval_mode = args.eval_mode
    save_path = args.save_path
    file_name = args.file_name

    """ Set evaluation envs
    """
    # set maze to be evaluated
    if len(args.maze_size_list) == 1:
        maze_size = [int(args.maze_size_list)]
    else:
        maze_size = [int(s) for s in args.maze_size_list.split(",")]
    if len(args.maze_seed_list) == 1:
        maze_seed = [int(args.maze_seed_list)]
    else:
        maze_seed = [int(s) for s in args.maze_seed_list.split(",")]
    if len(args.distance_list) == 1:
        distance_list = [int(args.distance_list)]
    else:
        distance_list = [int(s) for s in args.distance_list.split(",")]
    maze_size_eval = maze_size
    maze_seed_eval = maze_seed
    dist_eval = distance_list  # [2, 3, 6, 9, 10]

    """ Make environment
    """
    level_name = 'nav_random_maze_tile_bsp'
    # necessary observations (correct: this is the egocentric observations (following the counter clock direction))
    observation_list = [
        'RGB.LOOK_RANDOM_PANORAMA_VIEW',
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
                               use_true_state=args.use_true_state,
                               reward_type="sparse-1",
                               dist_epsilon=1e-3)

    """ Run different evaluation mode
    """
    if eval_mode == 'random-policy':
        # run the agent
        myVis = EvalPolicy(env=my_lab,
                           agent=None,
                           size_list=maze_size_eval,
                           seed_list=maze_seed_eval,
                           dist_list=dist_eval,
                           res_path=save_path,
                           res_name=file_name,
                           args=args)
        myVis.eval_random_policy()
    elif eval_mode == 'imagine-local-policy':
        my_agent = GoalDQNAgent(use_true_state=args.use_true_state, use_small_obs=True, use_rescale=args.use_rescale)
        # my_agent.policy_net.load_state_dict(
        #     torch.load(f"/mnt/cheng_results/results_RL/6-20/15-100-imagine/goal_ddqn_{15}x{15}_obs_100_imagine_seed_{1}.pt",
        #                map_location=torch.device('cpu'))
        # )
        my_agent.policy_net.load_state_dict(
            torch.load(f"./results/6-20/goal_ddqn_{15}x{15}_obs_50_imagine_seed_{1}.pt",
                       map_location=torch.device('cpu'))
        )
        
        my_agent.policy_net.eval()
        # run the agent
        myVis = EvalPolicy(env=my_lab,
                           agent=my_agent,
                           size_list=maze_size_eval,
                           seed_list=maze_seed_eval,
                           dist_list=dist_eval,
                           res_path=save_path,
                           res_name=file_name,
                           args=args)
        # myVis.eval_navigate_with_local_policy()
        # myVis.eval_navigate_with_local_policy_loop_entire()
        myVis.eval_navigate_with_hybrid_local_policy_using_dynamic_behavior_map()
    elif eval_mode == 'her-policy':
        my_agent = GoalDQNAgent(use_true_state=args.use_true_state, use_small_obs=True)
        my_agent.policy_net.load_state_dict(
            torch.load(f"/mnt/cheng_results/results_RL/6-20/baseline_2_random_goal_ddqn_{21}x{21}_obs_double_her_seed_1.pt",
                       map_location=torch.device('cpu'))
        )
        my_agent.policy_net.eval()
        # run the agent
        myVis = EvalPolicy(env=my_lab,
                           agent=my_agent,
                           size_list=maze_size_eval,
                           seed_list=maze_seed_eval,
                           dist_list=dist_eval,
                           res_path=save_path,
                           res_name=file_name,
                           args=args)
        myVis.eval_her_policy()


