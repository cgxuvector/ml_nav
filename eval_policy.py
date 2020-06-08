""" This script is a demo to control agent in a specific static maze environment in DeepMind Lab.
        - Evaluate baselines:
            - Random policy
            - Double DQN + HER
            - Map planner (TBD)
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
import matplotlib.pyplot as plt
import IPython.terminal.debugger as Debug

ACTION_LIST = ['up', 'down', 'left', 'right']


class EvalPolicy(object):
    def __init__(self, env, agent, size_list, seed_list, dist_list, res_path, res_name, args):
        # evaluation env object
        self.env = env
        self.device = args.device
        self.args = args
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
        self.use_imagine = True
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
        self.cvae.load_state_dict(torch.load("/mnt/cheng_results/trained_model/VAE/small_obs_L64_B8.pt", map_location=torch.device(self.device)))
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
                    print(f"Evaluate maze={m_size}-{m_seed} with dist={g_dist}")
                    # init the 3D environment
                    self.update_map2d_and_maze3d(set_new_maze=True, maze_size=m_size, maze_seed=m_seed, dist=g_dist)

                    # store evaluation results
                    run_num = self.run_num
                    success_count = 0
                    # start testing
                    for r in range(run_num):
                        # sample a random pair of start and goal positions
                        self.fix_start = False
                        self.fix_goal = False
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False,
                                                                                        maze_size=m_size,
                                                                                        maze_seed=m_seed,
                                                                                        dist=g_dist)
                        # set the maximal steps
                        print("Run idx = {}, Init = {}, Goal = {}, Dist = {}".format(r, start_pos, goal_pos, len(self.env_map.path)))
                        if self.args.step_type == 'optimal':
                            max_episode_len = (len(self.env_map.path) - 1)
                        else:
                            max_episode_len = 50
                        for t in range(max_episode_len):
                            # randomly sample an action
                            action = random.sample(range(4), 1)[0]
                            # step in the environment
                            next_state, reward, done, dist, trans, _, _ = my_lab.step(action)
                            
                            #print(f"state={state}, action={ACTION_LIST[action]}, next_state={next_state}, done={done}")
                            state = next_state
                            # check terminal
                            if done:
                                success_count += 1
                                #Debug.set_trace()
                                break
           
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

    def eval_navigate_with_local_policy(self):
        # store training data
        eval_results = defaultdict()
        # loop all the mazes
        for m_size in self.maze_size_list:
            for m_seed in self.maze_seed_list:
                # loop all the distance
                for g_dist in self.goal_dist:
                    print(f"Evaluate maze-{m_size}-{m_seed} with dist={g_dist}")
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
                        print("Run idx = {}, start pos = {}, goal pos = {}, dist = {}".format(r + 1, start_pos, goal_pos, distance))

                        # navigating between sub-goals
                        if not self.use_obs:
                            nav_sub_goals = sub_goals_pos
                        else:
                            nav_sub_goals = sub_goals_obs
                        state_pos = self.env.position_map2maze(start_pos + [0], [m_size, m_size])
                        for idx, g in enumerate(nav_sub_goals):
                            # flag for sub-goal navigation
                            sub_goal_done = False
                            # maximal steps for sub-goal navigation
                            if self.args.step_type == 'optimal':
                                max_time_step = 2
                            else:
                                max_time_step = 10
                            # convert the goal position to maze position
                            maze_goal_pos = self.env.position_map2maze(sub_goals_pos[idx], [m_size, m_size])
                            for t in range(max_time_step):
                                # get the action
                                action = self.agent.get_action(state, g, 0)
                                # save the action
                                act_list.append(ACTION_LIST[action])
                                # step the environment and print info
                                next_state, reward, done, dist, next_trans, _, _ = my_lab.step(action)
                                print(f"state={state_pos}, action={ACTION_LIST[action]}, next_state={next_trans}, done={done}")
                                state_pos = next_trans
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
                    if run_count:
                        print("Success rate = {}".format((run_count - fail_count) / run_count))
                        eval_results[f"{m_size}-{m_seed}-{g_dist}"] = (run_count - fail_count) / run_count
                    else:
                        print("No sampled pair is satisfied")
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
            maze_configs['update'] = False

        # obtain the state and goal observation
        state_obs, goal_obs, _, _ = self.env.reset(maze_configs)
        # return states and goals
        return state_obs, goal_obs, init_map_pos, goal_map_pos


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", type=str, default='random-policy', help="Evaluation mode")
    parser.add_argument("--eval_mode", type=str, default='train', help="test")
    parser.add_argument("--device", type=str, default='cpu', help="Device name")
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--maze_size_list", type=str, default='5', help="Maze size list")
    parser.add_argument("--maze_seed_list", type=str, default='0', help="Maze seed list")
    parser.add_argument("--dist_list", type=str, default="1", help="Distance list")
    parser.add_argument("--save_path", type=str, default='./', help="Save path")
    parser.add_argument("--file_name", type=str, default="random", help="File name")
    parser.add_argument("--run_num", type=int, default=100, help="Number of evaluation run")
    parser.add_argument("--use_true_state", type=str, default="True", help="Whether use the true state")
    parser.add_argument("--use_goal", type=str, default="True", help="Whether use the goal conditioned policy")
    parser.add_argument("--step_type", type=str, default="optimal", help="navigate to the goal in optimal steps") 
    parser.add_argument("--split_seed", type=int, default=0)
    
    return parser.parse_args()


# convert the True/False from str to bool
def strTobool(inputs):
    inputs.use_true_state = True if inputs.use_true_state == "True" else False
    inputs.use_goal = True if inputs.use_goal == "True" else False
    return inputs


if __name__ == '__main__':
    # set the evaluation model
    args = parse_input()
    args = strTobool(args)
    run_mode = args.run_mode
    eval_mode = args.eval_mode
    save_path = args.save_path
    file_name = args.file_name
    
    # random seed
    rnd_seed = args.random_seed
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)

    # set maze to be evaluated
    if len(args.maze_size_list) == 1:
        maze_size = [int(args.maze_size_list)]
    else:
        maze_size = [int(s) for s in args.maze_size_list.split(",")]
    if len(args.maze_seed_list) == 1:
        maze_seed = [int(args.maze_seed_list)]
    else:
        maze_seed = [int(s) for s in args.maze_seed_list.split(",")]
    if len(args.dist_list) == 1:
        dist_list = [int(args.dist_list)]
    else:
        dist_list = [int(s) for s in args.dist_list.split(",")]
    maze_size_eval_trn = maze_size
    maze_seed_eval_trn = maze_seed
    maze_size_eval_tst = list(set(range(5, 23, 2)) - set(maze_size_eval_trn))
    maze_seed_eval_tst = list(set(range(0, 20, 1)) - set(maze_seed_eval_trn))
    dist_eval = dist_list
    
    # compute the train mazes
    if eval_mode == "train":
        maze_size_eval = maze_size_eval_trn
        maze_seed_eval = maze_seed_eval_trn
    else:
        maze_size_eval = maze_size_eval_tst
        maze_seed_eval = maze_seed_eval_tst

    # set level name
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

    if run_mode == 'random-policy':
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
    elif run_mode == 'imagine-local-policy':
        my_agent = GoalDQNAgent(use_true_state=args.use_true_state, use_small_obs=True)
        my_agent.policy_net.load_state_dict(
            torch.load(f"/mnt/cheng_results/results_RL/5-30/random_imagine_goal_ddqn_{7}x{7}_obs_double_random_maze_env2_seed_{args.split_seed}.pt",
                map_location=torch.device(args.device))
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
        myVis.eval_navigate_with_local_policy()
    else:
        raise Exception(f"Invalid run mode. Expect random-policy or imagine-local-policy, but get {args.run_mode}")
