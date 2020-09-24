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
from envs.LabEnvV4 import RandomMaze
import numpy as np
import pickle
from utils import searchAlg
import IPython.terminal.debugger as Debug
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ACTION_LIST = ['up', 'down', 'right', 'left']


class EvalPolicy(object):
    def __init__(self, env, agent, size_list, seed_list, dist_list, res_path, res_name, args):
        # set the device
        self.device = torch.device(args.device)
        # evaluation env object
        self.env = env
        # 2D rough map object
        self.env_map = None
        # agent object
        self.agent = agent
        # mazes to be evaluated
        # self.maze_size = args.model_maze_size
        self.maze_size = 0
        self.maze_size_list = size_list
        self.maze_seed_list = seed_list
        # evaluation protocol
        self.fix_start = False
        self.fix_goal = False
        self.theme_list = ['MISHMASH']
        self.decal_list = [0.001]
        self.use_goal = None
        self.use_true_state = args.use_true_state
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
        # self.cvae.load_state_dict(torch.load("/mnt/cheng_results/VAE/models/small_obs_L64_B8.pt", map_location=torch.device('cuda:0')))
        self.cvae.load_state_dict(torch.load("./results/vae/model/small_obs_L64_B8.pt", map_location=torch.device('cpu')))
        # self.cvae.load_state_dict(torch.load("/mnt/sda/dataset/ml_nav/VAE/model/small_obs_L64_B8.pt", map_location=torch.device('cpu')))
        self.cvae.eval()
        self.cvae = self.cvae.to(self.device)
        # save parameters
        self.save_path = res_path
        self.file_name = res_name

        # maximal episode length
        self.max_episode_len = args.max_episode_len
        self.args = args

    def eval_random_policy(self):
        # save the data
        eval_results = defaultdict()
        # loop all the mazes
        for m_size in self.maze_size_list:
            for m_seed in self.maze_seed_list:
                # load all possible pairs
                total_pairs_dict = self.load_pair_data(m_size, m_seed)
                pairs_dict = {'start': None, 'goal': None}
                # loop all distances
                for g_dist in self.goal_dist:
                    # check the distance validation
                    if not str(g_dist) in total_pairs_dict.keys():
                        print(f"Maze {m_size}-{m_seed} has no pair with distance {g_dist}")
                        break
                    # get the start and goal pairs
                    pairs_dict['start'] = total_pairs_dict[str(g_dist)][0]
                    pairs_dict['goal'] = total_pairs_dict[str(g_dist)][1]
                    total_sub_pair_num = len(pairs_dict['start'])
                    # init the 3D environment
                    self.update_map2d_and_maze3d(set_new_maze=True, maze_size=m_size, maze_seed=m_seed, dist=g_dist)
                    # store evaluation results
                    run_num = self.run_num
                    success_count = 0
                    run_count = 0
                    # start testing
                    for r in range(run_num):
                        # sample a random pair of start and goal positions
                        pair_idx = random.sample(range(total_sub_pair_num), 1)[0]
                        if random.uniform(0, 1) < 0.5:
                            s_pos = pairs_dict['start'][pair_idx]
                            g_pos = pairs_dict['goal'][pair_idx]
                        else:
                            s_pos = pairs_dict['goal'][pair_idx]
                            g_pos = pairs_dict['start'][pair_idx]

                        # set the environment
                        state, goal, start_pos, goal_pos = self.update_maze_from_pos(start_pos=s_pos, goal_pos=g_pos)
                        
                        # start random navigation
                        max_episode_len = self.max_episode_len
                        for t in range(max_episode_len):
                            # randomly sample an action
                            action = random.sample(range(4), 1)[0]
                            # step in the environment
                            next_state, reward, done, dist, trans, _, _ = my_lab.step(action)

                            # check terminal
                            if done:
                                success_count += 1
                                break
                        print(f"run {r}: {m_size}-{m_seed}: Start = {s_pos}, Goal = {g_pos}, Dist = {g_dist}, Done = {done}")

                    
                    # print the results
                    print("Success rate = {}".format(success_count / run_num))
                    # store the results
                    eval_results[f"{m_size}-{m_seed}-{g_dist}"] = success_count / run_num
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
                # load all possible pairs
                total_pairs_dict = self.load_pair_data(m_size, m_seed)
                pairs_dict = {'start': None, 'goal': None}
                # loop all distances
                for g_dist in self.goal_dist:
                    # check the distance validation
                    if not str(g_dist) in total_pairs_dict.keys():
                        print(f"Maze {m_size}-{m_seed} has no pair with distance {g_dist}")
                        break
                    # get the start and goal pairs
                    pairs_dict['start'] = total_pairs_dict[str(g_dist)][0]
                    pairs_dict['goal'] = total_pairs_dict[str(g_dist)][1]
                    total_sub_pair_num = len(pairs_dict['start'])
                    # init the 3D environment
                    self.update_map2d_and_maze3d(set_new_maze=True, maze_size=m_size, maze_seed=m_seed, dist=g_dist)
                    # store evaluation results
                    run_num = self.run_num
                    success_count = 0
                    run_count = 0
                    # start testing
                    for r in range(run_num):
                        # sample a random pair of start and goal positions
                        pair_idx = random.sample(range(total_sub_pair_num), 1)[0]
                        if random.uniform(0, 1) < 0.5:
                            s_pos = pairs_dict['start'][pair_idx]
                            g_pos = pairs_dict['goal'][pair_idx]
                        else:
                            s_pos = pairs_dict['goal'][pair_idx]
                            g_pos = pairs_dict['start'][pair_idx]

                        # set the environment
                        state, goal, start_pos, goal_pos = self.update_maze_from_pos(start_pos=s_pos, goal_pos=g_pos)
                        max_episode_len = 100
                        run_count += 1
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
                        print(f"run {r}: {m_size}-{m_seed}: Start = {s_pos}, Goal = {g_pos}, Dist = {g_dist}, Done = {done}")

                    # print the results
                    print("Success rate = {}".format(success_count / run_count))
                    # store the results
                    eval_results[f"{m_size}-{m_seed}-{g_dist}"] = success_count / run_count
        # print info
        print("Evaluation finished")
        # save the dictionary as txt file
        save_name = '/'.join([self.save_path, f"eval_{self.file_name}_policy.txt"])
        with open(save_name, 'w') as f:
            for key, val in eval_results.items():
                tmp_str = key + ' ' + str(val) + '\n'
                f.write(tmp_str)
            f.close()
    
    def eval_map_1_step_navigation(self):
        # find all size and seed
        for m_size in self.maze_size_list:
            for m_seed in self.maze_seed_list:
                print(f"size={m_size}, seed={m_seed}")
                # load total pairs
                total_pairs_dict = self.load_pair_data(m_size, m_seed)
                pairs_dict = {'start': [], 'goal': []}
                self.update_map2d_and_maze3d(set_new_maze=True, maze_size=m_size, maze_seed=m_seed, dist=-1)
                print(f"{self.goal_dist}")
                # extract particular distance
                for g_dist in self.goal_dist:
                    pairs_dict['start'] = total_pairs_dict[str(g_dist)][0]
                    pairs_dict['goal'] = total_pairs_dict[str(g_dist)][1]
                    # success count
                    success_run = 0
                    # total count 
                    total_run = 0
                    # success flag
                    success_flag = False
                    # loop all possible pairs
                    for s_pos, g_pos in zip(pairs_dict['start'], pairs_dict['goal']):
                        state, goal, start_pos, goal_pos = self.update_maze_from_pos(s_pos, g_pos)
                        max_time_steps = 200
                        state_pos_maze = self.env.position_map2maze(start_pos + [0], [m_size, m_size])
                        goal_pos_maze = self.env.position_map2maze(goal_pos + [0], [m_size, m_size])
                        # whether use the real observation or the fake
                        if not self.use_true_state and self.use_imagine:
                            goal_loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded, goal_pos)
                            goal = self.imagine_goal_obs(goal_loc_map)
                        # test one pair
                        for t in range(max_time_steps):
                            # get the action
                            action = self.agent.get_action(state, goal, 0)
                            # take the next step
                            next_state, reward, done, dist, next_trans, _, _, _, _ = self.env.step(action)
                            
                            print(f"Step = {t}: state={state_pos_maze}, act={ACTION_LIST[action]}, next_state={next_trans}, goal={goal_pos_maze}, dist={dist}, done={done}")

                            if done:
                                success_run += 1
                                success_flag = True
                                print(f"Time step = {t}")
                                break
                            else:
                                state = next_state
                                state_pos_maze = next_trans
                        total_run += 1
                        print(f"run idx={total_run}: start={s_pos}, goal={g_pos}, done={done}, success={success_flag}") 
                        
                        success_flag = False
                        state, goal, start_pos, goal_pos = self.update_maze_from_pos(g_pos, s_pos)
                        max_time_steps = 50
                        state_pos_maze = self.env.position_map2maze(start_pos + [0], [m_size, m_size])
                        goal_pos_maze = self.env.position_map2maze(goal_pos + [0], [m_size, m_size])
                        # whether use the real observation or the fake
                        if not self.use_true_state and self.use_imagine:
                            goal_loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded, goal_pos)
                            goal = self.imagine_goal_obs(goal_loc_map)
                        # test one pair
                        for t in range(max_time_steps):
                            # get the action
                            action = self.agent.get_action(state, goal, 0)
                            # take the next step
                            next_state, reward, done, dist, next_trans, _, _, _, _ = self.env.step(action)

                            print(
                                f"state={state_pos_maze}, act={ACTION_LIST[action]}, next_state={next_trans}, goal={goal_pos_maze}, dist={dist}, done={done}")
                            if done:
                                success_run += 1
                                success_flag = True
                                break
                            else:
                                state = next_state
                                state_pos_maze = next_trans
                        total_run += 1
                        print(f"run idx={total_run}: start={g_pos}, goal={s_pos}, done={done}, success={success_flag}")
                        success_flag = False
                        print('--------------------------------------------------------------------------------------')
                    print(f"Total pairs={total_run}, Success pairs={success_run}, Success rate={success_run/total_run}")

    def eval_navigate_with_dynamic_topological_map(self):
        # store the evaluation results in a dictionary
        eval_results = defaultdict()
        # loop all evaluation mazes
        for m_size in self.maze_size_list:
            self.maze_size = m_size
            for m_seed in self.maze_seed_list:
                # initialize lab the environment
                print(f"Init maze {m_size}-{m_seed} environment")
                self.update_map2d_and_maze3d(set_new_maze=True, maze_size=m_size, maze_seed=m_seed, dist=-1)
                # build the initial dynamic behavior map
                if self.args.imprecise_rate > 0:
                    # build the map using imprecise 2d map
                    self.env_map.shuffle_map(args.imprecise_rate, 'mixed')
                    init_mlb_map = self.build_mlb_from_2d_imprecise_map(max_edge_len=1)
                else:
                    init_mlb_map = self.build_mlb_from_2d_map(max_edge_len=1)
                    np.save(f"./mlb_map_{m_size}_{m_seed}.npy", init_mlb_map)
                    # build the map using rough 2-D map
                    # init_mlb_map = np.load(f'/mnt/cheng_results/mlb_map_{m_size}_{m_seed}.npy')
                    # init_mlb_map = np.load(f'/mnt/sda/mlb_map_{m_size}_{m_seed}.npy')
                # obtain all the distance pairs
                total_pairs_dict = self.load_pair_data(m_size, m_seed)
                pairs_dict = {'start': None, 'goal': None}
                # loop all possible distance 
                for g_dist in self.goal_dist:
                    # init the map for different distance
                    mlb_map = init_mlb_map.copy()
                    mlb_graph = csr_matrix(mlb_map)
                    # check if the distance is valid
                    if not str(g_dist) in total_pairs_dict.keys():
                        print(f"Maze {m_size}-{m_seed} has no pair with distance {g_dist}")
                        eval_results[f"{m_size}-{m_seed}-{g_dist}"] = None
                        break
                    pairs_dict['start'] = total_pairs_dict[str(g_dist)][0]
                    pairs_dict['goal'] = total_pairs_dict[str(g_dist)][1]
                    total_sub_pair_num = len(pairs_dict['start'])
                    # evaluate counter
                    success_counter = 0
                    total_runs = (total_sub_pair_num * 2) if (total_sub_pair_num * 2) <= 100 else 100
                    # loop all possible pairs
                    print(f"maze: {m_size}-{m_seed}-{g_dist}")
                    valid_run_count = 0
                    total_runs = len(pairs_dict['start'])
                    total_runs = args.run_num
                    run = 0
                    pos_unconnected_num = 0
                    pos_single_connected_num = 0
                    for s_pos, g_pos in zip(pairs_dict['start'], pairs_dict['goal']):
                        print(s_pos, ' - ', g_pos)
                    #run = 0
                    #while run < total_runs:
                        # sample a start-goal pair
                        pair_idx = random.sample(range(total_sub_pair_num), 1)[0]
                        # with probability 0.5, reverse the order
                        #if random.uniform(0, 1) < 0.5:
                        #    s_pos = pairs_dict['start'][pair_idx]
                        #    g_pos = pairs_dict['goal'][pair_idx]
                        #else:
                        #    s_pos = pairs_dict['goal'][pair_idx]
                        #    g_pos = pairs_dict['start'][pair_idx]
                        # option for imprecise maze
                        #if self.args.imprecise_rate > 0:
                        #    if not s_pos in self.env_map.imprecise_valid_pos or not g_pos in self.env_map.imprecise_valid_pos:
                        #        print(f"{s_pos} or {g_pos} is not in imprecise valid positions")
                        #        continue
                        
                        # get the index of the start and goal positions 
                        if g_dist == 1:
                            s_idx = self.env_map.valid_pos.index(s_pos)
                            g_idx = self.env_map.valid_pos.index(g_pos)
                        
                        # forward
                        success_flag_forward = self.run_single_pair(s_pos, g_pos, mlb_map, mlb_graph)
                        if success_flag_forward:
                            success_counter += 1
                            # update the graph only for dist = 1
                            if g_dist == 1:
                                if mlb_map[s_idx, g_idx] == 0:
                                    mlb_map[s_idx, g_idx] = 1
                                    mlb_graph = csr_matrix(mlb_map)
                        run += 1 
                        print(f"{m_size}-{m_seed}: Run {run}: Start = {s_pos}, Goal = {g_pos}, Dist = {len(self.env_map.path)}, Done = {success_flag_forward}")  
                        # Debug.set_trace()

                        # backward
                        success_flag_backward = self.run_single_pair(g_pos, s_pos, mlb_map, mlb_graph)
                        if success_flag_backward:
                            success_counter += 1
                            if g_dist == 1:
                                if mlb_map[g_idx, s_idx] == 0:
                                    mlb_map[g_idx, s_idx] = 1
                                    mlb_graph = csr_matrix(mlb_map)
                        run += 1
                            
                        print(f"{m_size}-{m_seed}: Run {run}: Start = {g_pos}, Goal = {s_pos}, Dist = {len(self.env_map.path)}, Done = {success_flag_backward}") 
                        # reset flag
                        success_flag_forward = False
                        success_flag_backward = False
                        print(f"--------------------------------------------------------------------")
                        
                    # show results
                    #np.save(f'./eval_results/novel_mazes/{m_size}-{self.args.model_maze_seed}-{m_seed}_tdm.npy', mlb_map)
                    #np.save(f'./eval_results/novel_mazes/{m_size}-{self.args.model_maze_seed}-{m_seed}-True-valid-pos.npy', self.env_map.valid_pos)
                    #np.save(f'./eval_results/novel_mazes/{m_size}-{self.args.model_maze_seed}-{m_seed}-success.npy', [float(success_counter)/run, run])
                    #np.save(f'{m_size}-{m_seed}-Fail-pos.npy', fail_cases)
                    print(f"Success count = {success_counter}, Total count = {run}")
                    print(f"Single connected positons = {pos_single_connected_num}, Unconnected positions = {pos_unconnected_num}")
                    print(f"Mean successful rate for distance = {g_dist} is {success_counter / run}")
                    
                    eval_results[f"{m_size}-{m_seed}-{g_dist}"] = float(success_counter) / run
   
                # print info
                #print("Evaluation finished")
                # save the dictionary as txt file
                #save_name = '/'.join([self.save_path, f"eval_{self.file_name}_policy.txt"])
                #with open(save_name, 'w') as f:
                #    for key, val in eval_results.items():
                #        tmp_str = key + ' ' + str(val) + '\n'
                #        f.write(tmp_str)
                #    f.close()
                
    def run_single_pair(self, s_pos, g_pos, mlb_map, mlb_graph):
        act_list = []
        act_idx_list = []
        # success flag
        success_flag = False 
        # set the environment based on the sampled start and goal positions
        start_obs, goal_obs, start_pos, goal_pos = self.update_maze_from_pos(start_pos=s_pos,
                                                                             goal_pos=g_pos)
        # set the budget for the navigation
        max_time_steps = self.max_episode_len
        state_pos = start_pos
        state_obs = start_obs
        last_way_point_obs = start_obs
        # plan for every time steps
        t_counter = 0
        nav_done = False
        fail_count = 0
        current_pos_map = state_pos
        goal_pos_maze = self.env.position_map2maze(goal_pos + [0], [self.maze_size, self.maze_size])
        state_pos_maze = self.env.position_map2maze(state_pos + [0], [self.maze_size, self.maze_size])
        # start one navigation
        while t_counter < max_time_steps:  
            # get the next way point: observation, position (map)
            next_landmark_obs, next_landmark_pos_map = self.mlb_search_next_waypoint(state_pos, goal_pos, mlb_graph) 
            
            # check whether there is a connected path
            if next_landmark_obs is None or next_landmark_pos_map is None:
                print(f"None path is found between {state_pos} and {goal_pos}")
                success_flag = False
                break

            # compute the pos in the environment to decide the sub-goal termination
            next_landmark_pos_maze = self.env.position_map2maze(next_landmark_pos_map + [0], [self.maze_size, self.maze_size])
            
            # way point navigation budget
            max_steps_per_goal = 10 # steps for each landmark
            sub_nav_done = False
            sub_action_list = []
            last_landmark_obs = state_obs
            while max_steps_per_goal > 0:
                # get the action
                action = self.agent.get_action(state_obs, next_landmark_obs, 0)
                # agent take that action
                next_state, reward, done, dist, next_trans, _, _, _, _ = my_lab.step(action)
                # record the feedback
                sub_action_list.append(action)
                act_idx_list.append(action)
                
                # for pred-variant
                #with torch.no_grad():
                #    current_state = self.toTensor(next_state) / 255
                #    target_goal = self.toTensor(next_landmark_obs) / 255
                #    _, state_pred = self.agent.policy_net(current_state, target_goal) 
                #    reach_sub_goal = state_pred.view(1, -1).max(dim=1)[1].item()
                print(f"State={state_pos_maze}, Action={ACTION_LIST[action]}, Next state={next_trans}, Reward={reward}, Goal={next_landmark_pos_maze}, dist={dist:.2f}, Done={done}")
                #Debug.set_trace()
                # update the current state
                nav_done = done
                state_obs = next_state
                state_pos_maze = next_trans
                #Debug.set_trace() 
                # increase and decrease the indicator
                t_counter += 1
                max_steps_per_goal -= 1
                
                # check terminal
                if args.use_oracle:
                    sub_goal_dist = self.compute_distance(next_trans, next_landmark_pos_maze)
                    if sub_goal_dist <= args.terminal_dist:
                        print(f'Reach the way point {next_landmark_pos_map} with action = {ACTION_LIST[action]}')
                        sub_nav_done = True
                        act_list.append(ACTION_LIST[action])
                        break
                #else:
                #    if reach_sub_goal == 1:
                #        print(f'reach waypoint {next_landmark_pos_map}')
                #        sub_nav_done = True
                #        act_list.append(ACTION_LIST[action])
                #        break 

            # if agent fail to reach the way point
            if not sub_nav_done:
                print(f"Fail to reach the way point {next_landmark_pos_map}")
                # Debug.set_trace()
                fail_count += 1
                target_position_maze = self.env.position_map2maze(state_pos + [0], [self.maze_size, self.maze_size])
                for action in range(len(sub_action_list)):
                    # estimate the current position
                    #with torch.no_grad():
                    #    _, state_pred = self.agent.policy_net(self.toTensor(state_obs)/255, self.toTensor(last_way_point_obs)/255)
                    #    reach_sub_goal = state_pred.view(1, -1).max(dim=1)[1].item() 
                    
                    # check terminal
                    if args.use_oracle:
                        sub_goal_dist = self.compute_distance(next_trans, target_position_maze)
                        if sub_goal_dist <= args.terminal_dist:
                            print(f"return to the last waypoint {state_pos}")
                            break
                    #else:
                    #    if reach_sub_goal == 1:
                    #        print(f'return to the last waypoint {state_pos}')
                    #        sub_nav_done = True
                    #        break

                    # Debug.set_trace()
                    #reversed_act = reversed_actions[action]
                    reversed_act = self.agent.get_action(state_obs, last_landmark_obs, 0)
                    act_idx_list.append(reversed_act)
                    # take the step
                    next_state, reward, _, dist, next_trans, _, _, _, _ = my_lab.step(reversed_act)
                   
                    # print information
                    #print(f"Return: state={self.env.position_maze2map(state_pos_maze, [self.maze_size, self.maze_size])}, Action={ACTION_LIST[reversed_act]}, next state={self.env.position_maze2map(next_trans, [self.maze_size, self.maze_size])}, goal = {self.env.position_maze2map(target_position_maze, [self.maze_size, self.maze_size])}")
                    #Debug.set_trace()
                    state_pos_maze = next_trans
                    state_obs = next_state
                    t_counter += 1
                    
                    # estimate the current position
                    #with torch.no_grad():
                    #    _, state_pred = self.agent.policy_net(self.toTensor(state_obs)/255, self.toTensor(last_way_point_obs)/255)
                    #    reach_sub_goal = state_pred.view(1, -1).max(dim=1)[1].item() 

                    # check terminal
                    if args.use_oracle:
                        sub_goal_dist = self.compute_distance(next_trans, target_position_maze)
                        if sub_goal_dist <= args.terminal_dist:
                            print(f"return to the last waypoint {state_pos}")
                            break
                    #else:
                    #    if reach_sub_goal == 1:
                    #        print(f'return to the last waypoint {state_pos}')
                    #        sub_nav_done = True
                    #        break

                # update the graph
                state_idx = self.env_map.valid_pos.index(state_pos) if not self.args.imprecise_rate > 0 else self.env_map.imprecise_valid_pos.index(state_pos)
                goal_idx = self.env_map.valid_pos.index(next_landmark_pos_map) if not self.args.imprecise_rate > 0 else self.env_map.imprecise_valid_pos.index(next_landmark_pos_map)
                # based on the observation variance, I will let the agent try several times
                if fail_count > 10:
                    print("Update graph")
                    mlb_map[state_idx, goal_idx] = 0
                    fail_count = 0
                    mlb_graph = csr_matrix(mlb_map)
                # if it fails for twice, I will use the imagination to guess an action
                if fail_count > 8:
                    print("Use fake current obs")
                    if not self.args.imprecise_rate > 0:
                        loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded, state_pos)
                    else:
                        loc_map = self.env_map.cropper(self.env_map.map2d_roughpad_imprecise, state_pos)
                    state_obs_imagined = self.imagine_goal_obs(loc_map)
                    state_obs = state_obs_imagined
            else:
                state_pos = next_landmark_pos_map

            # check the final terminal
            if nav_done:
                success_flag = True
                #if len(act_list) > len(self.env_map.path):
                #    print(act_idx_list)
                #    Debug.set_trace()
                break

            if not nav_done and state_pos == goal_pos:
                success_flag = False
                state_idx = self.env_map.valid_pos.index(state_pos) if not self.args.imprecise_rate > 0 else self.env_map.imprecise_valid_pos.index(state_pos)
                goal_idx = self.env_map.valid_pos.index(next_landmark_pos_map) if not self.args.imprecise_rate > 0 else self.env_map.imprecise_valid_pos.index(next_landmark_pos_map)
                mlb_map[state_idx, goal_idx] = 0
                print("Wrong waypoint reaching estimation")
                break
        return success_flag

    def test_distance_estimation(self):
        state, goal, state_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=True, dist=self.goal_dist)
        run_num = 50
        for r in range(run_num):
            gt_dist = len(self.env_map.path) - 1
            
            loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded, goal_pos)
            goal_obs_imagined = self.imagine_goal_obs(loc_map)
            goal_obs = self.toTensor(goal_obs_imagined) / 255

            with torch.no_grad():
                state = self.toTensor(state)/255
                goal = self.toTensor(goal)/255
                pred_dist = self.agent.policy_net(goal, goal_obs)

            pred_distance= pred_dist.max()
            action = pred_dist.view(1, -1).max(dim=1)[1].item() 
            print(f"State = {state_pos}, Goal = {goal_pos}, Act = {ACTION_LIST[action]} GT = {gt_dist}, Pred = {-1 * pred_distance}, Err={gt_dist - (-1*pred_distance)}")

            self.fix_start = False
            self.fix_goal = False
            state, goal, state_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False, dist=self.goal_dist)
                
    # search for the action using mlb
    def mlb_search_next_waypoint(self, state_pos, goal_pos, mlb_graph):
        # get next way point
        next_waypoint_pos = self.mlb_get_next_waypoint(state_pos, goal_pos, mlb_graph)
        if next_waypoint_pos != -9999:
            # generate next way point observation
            if not self.args.imprecise_rate > 0:
                waypoint_loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded, next_waypoint_pos)
            else:
                waypoint_loc_map = self.env_map.cropper(self.env_map.map2d_roughpad_imprecise, next_waypoint_pos)
            next_waypoint_obs = self.imagine_goal_obs(waypoint_loc_map)
            return next_waypoint_obs, next_waypoint_pos
        else:
            return None, None

    def mlb_get_next_waypoint(self, state_pos, goal_pos, graph): 
        # get the position indices
        if not self.args.imprecise_rate > 0:
            state_idx = self.env_map.valid_pos.index(state_pos)
            goal_idx = self.env_map.valid_pos.index(goal_pos)
        else:
            state_idx = self.env_map.imprecise_valid_pos.index(state_pos)
            goal_idx = self.env_map.imprecise_valid_pos.index(goal_pos)
        # search for the shortest path using Dijkstra's algorithm
        dist_matrix, predecessors = dijkstra(csgraph=graph, directed=True, indices=state_idx, return_predecessors=True)
        # generate the path
        path = []
        idx = goal_idx
        while idx != state_idx:
            if not self.args.imprecise_rate > 0:
                path.append(self.env_map.valid_pos[idx])
            else:
                path.append(self.env_map.imprecise_valid_pos[idx])
            idx = predecessors[idx]
            if idx == -9999:
                break

        # deal with the case when there is no path
        if idx != -9999:
            next_waypoint_pos = path[-1]
            if not self.args.imprecise_rate > 0:
                path.append(self.env_map.valid_pos[state_idx])
            else:
                path.append(self.env_map.imprecise_valid_pos[state_idx])
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

    def build_mlb_from_2d_imprecise_map(self, max_edge_len=1):
        # get all the valid positions on the map
        map_valid_pos = self.env_map.imprecise_valid_pos
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
                tmp_path = searchAlg.A_star(self.env_map.map2d_imprecise, s_pos, g_pos)
                # we only consider few steps away

                if tmp_path is None or len(tmp_path) > max_dist:
                    continue
                # assume fully connectivity
                print(f"Start = {s_pos}, Goal = {g_pos}, Distance = {len(tmp_path)}")
                mlb_sparse_matrix[i, j] = len(tmp_path) - 1
        return mlb_sparse_matrix

    def load_pair_data(self, m_size, m_seed):
        # path = f'/mnt/cheng_results/map/maze_{m_size}_{m_seed}.pkl'
        path = f'./map/maze_{m_size}_{m_seed}.pkl'
        # path = f'/mnt/sda/map/maze_{m_size}_{m_seed}.pkl'
        f = open(path, 'rb')
        return pickle.load(f)

    def compute_distance(self, source_pos, target_pos):
        dist = np.sqrt((source_pos[0] - target_pos[0]) ** 2 + (source_pos[1] - target_pos[1]) ** 2)
        return dist

    def toTensor(self, obs_list):
        if not self.use_true_state:
            state_obs = torch.tensor(np.array(obs_list).transpose(0, 3, 1, 2), dtype=torch.float32, device=self.device)
        else:
            state_obs = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        return state_obs

    def imagine_goal_obs(self, pos_loc_map):
        imagined_obs = []
        loc_map = torch.from_numpy(pos_loc_map).flatten().view(1, -1).float()
        for ori in self.orientations:
            z = torch.randn(1, 64)
            tmp_map = torch.cat(2 * [loc_map], dim=1)
            tmp_ori = torch.cat(2 * [ori.view(-1, 1 * 1 * 8).float()], dim=1)
            conditioned_z = torch.cat((z, tmp_map, tmp_ori), dim=1)
            conditioned_z = conditioned_z.to(self.device)
            obs_reconstructed, _ = self.cvae.decoder(conditioned_z)
            obs_reconstructed = obs_reconstructed.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0) * 255
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
        state_obs, goal_obs, _, _, _, _ = self.env.reset(maze_configs)
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
        state_obs, goal_obs, _, _, _, _ = self.env.reset(maze_configs)
        # return states and goals
        return state_obs, goal_obs, start_pos, goal_pos


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_mode", type=str, default='random-policy', help="Evaluation mode: random-policy, "
                                                                               "her-policy, or our-policy")
    parser.add_argument("--maze_size_list", type=str, default='5', help="Maze size list")
    parser.add_argument("--maze_seed_list", type=str, default='0', help="Maze seed list")
    parser.add_argument("--model_maze_size", type=int, default=5, help="Model maze size")
    parser.add_argument("--model_maze_seed", type=int, default=0, help="Model seed size")
    parser.add_argument("--model_epoch", type=int, default=0, help='Model epoch')
    parser.add_argument("--model_dist", type=int, default=1, help='Model distance')
    parser.add_argument("--distance_list", type=str, default="1", help="Distance list")
    parser.add_argument("--save_path", type=str, default='./eval_results', help="Save path")
    parser.add_argument("--file_name", type=str, default="test", help="File name")
    parser.add_argument("--run_num", type=int, default=100, help="Number of evaluation run")
    parser.add_argument("--use_true_state", action='store_true', default=False, help="Whether use the true state")
    parser.add_argument("--use_goal", action='store_true', default=False, help="Whether use the goal conditioned policy")
    parser.add_argument("--use_imagine", action='store_true', default=False, help="Whether use the imagined observation")
    parser.add_argument("--use_state_est", action='store_true', default=False, help="Whether use the state estimation")
    parser.add_argument("--use_rescale", action='store_true', default=False, help="Whether use the rescaled observation")
    parser.add_argument("--max_episode_len", type=int, default=100, help="Max time steps per episode")
    parser.add_argument("--seed_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default='cuda:0', help="device")   
    parser.add_argument("--use_oracle", action='store_true', default=False, help='whether use the orcale to decide waypoint')
    parser.add_argument("--random_seed", type=int, default='1234', help="random seed")
    parser.add_argument("--imprecise_rate", type=float, default=-1, help="imprecise rate of the 2d map")
    parser.add_argument("--terminal_dist", type=float, default=4, help='terminal distance')
    
    return parser.parse_args() 


if __name__ == '__main__':
    # set the evaluation model
    args = parse_input()
    eval_mode = args.eval_mode
    save_path = args.save_path
    file_name = args.file_name

    # random seed & device
    rnd_seed = args.random_seed
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed) 
    device = torch.device(args.device)
        
    """ Set evaluation envs
    """
    # set maze to be evaluated
    maze_size = [int(s) for s in args.maze_size_list.split(",")]
    maze_seed = [int(s) for s in args.maze_seed_list.split(",")]
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
        'RGB.LOOK_TOP_DOWN_VIEW',
        'RGB.LOOK_PANORAMA_VIEW',
        'DEBUG.POS.TRANS',
        'DEBUG.POS.ROT',
        'VEL.TRANS',
        'VEL.ROT'
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
    my_lab = RandomMaze(level_name,
                        observation_list,
                        configurations,
                        use_true_state=args.use_true_state,
                        reward_type="sparse-1",
                        dist_epsilon=args.terminal_dist)

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
    elif eval_mode == 'our-policy':
        my_agent = GoalDQNAgent(use_true_state=args.use_true_state,
                                use_small_obs=True,
                                use_rescale=args.use_rescale,
                                device=device,
                                use_state_est=args.use_state_est)
        # my_agent.policy_net.load_state_dict(torch.load(f"/mnt/cheng_results/rl_results/corl/one_shot/{args.model_maze_size}-1/{args.model_maze_seed}/goal_dqn_vanilla_soft_local_obs_{args.model_maze_size}x{args.model_maze_size}_obs_dist_{args.model_dist}_68001.pt", map_location=torch.device(args.device))
        # )
        # print(f"/mnt/cheng_results/rl_results/corl/one_shot/{args.model_maze_size}-1/{args.model_maze_seed}/goal_dqn_vanilla_soft_local_obs_{args.model_maze_size}x{args.model_maze_size}_obs_dist_{args.model_dist}.pt")
        my_agent.policy_net.load_state_dict(torch.load(f"./results/9-23/goal_ddqn_obs_{args.model_maze_size}x{args.model_maze_size}_obs_dist_{args.model_dist}.pt", map_location=torch.device(args.device))
        )
        #my_agent.policy_net.load_state_dict(torch.load(
        #    f"./one_shot/{args.model_maze_size}--1/{args.model_maze_seed}/true_state_test_ep_100_{args.model_maze_size}x{args.model_maze_size}_obs_dist_-1.pt",
        #    map_location=torch.device(args.device))
        #)
        # my_agent.policy_net.load_state_dict(
        #         torch.load(f'/mnt/sda/rl_results/corl/one_shot/{args.model_maze_size}-1/{args.model_maze_seed}/goal_ddqn_obs_{args.model_maze_size}x{args.model_maze_size}_obs_dist_1.pt', map_location=torch.device(args.device))
        # )
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
        myVis.eval_navigate_with_dynamic_topological_map()
        # myVis.eval_map_1_step_navigation()
    elif eval_mode == 'her-policy':
        my_agent = GoalDQNAgent(use_true_state=args.use_true_state,
                                use_small_obs=True,
                                device=device)
        my_agent.policy_net.load_state_dict(
            torch.load(f"/mnt/cheng_results/rl_results/7-7/{args.model_maze_size}-{args.model_maze_size}/{args.model_maze_seed}/goal_ddqn_HER_{args.model_maze_size}x{args.model_maze_size}_obs.pt", map_location=args.device)
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
