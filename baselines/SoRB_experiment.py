from utils import mapper
import numpy as np
from collections import namedtuple, defaultdict
import tqdm
from model import VAE
from utils import memory
from utils import ml_schedule
from utils import searchAlg
import torch
import random
import os
import numpy as np
import time
import scipy.sparse
import pickle
import IPython.terminal.debugger as Debug

# define the global default parameters
DEFAULT_TRANSITION = namedtuple("transition", ["state", "action", "next_state", "reward", "goal", "done"])
DEFAULT_ACTION_LIST = ['up', 'down', 'left', 'right']


class Experiment(object):
    def __init__(self,
                 env,  # environment configurations
                 agent,
                 maze_list,
                 seed_list,
                 dist_list,
                 decal_freq=0.1,
                 fix_maze=True,
                 fix_start=True,
                 fix_goal=True,
                 use_goal=False,
                 goal_dist=np.inf,
                 max_dist=4,
                 use_true_state=False,  # state configurations
                 sample_start_goal_num=10,
                 train_episode_num=10,  # training configurations
                 start_train_step=1000,
                 max_time_steps=50000,
                 episode_time_steps=100,
                 use_replay=False,
                 use_her=False,  # HER configurations
                 future_k=4,
                 buffer_size=20000,
                 transition=DEFAULT_TRANSITION,
                 learning_rate=1e-3,  # optimization configurations
                 batch_size=64,
                 gamma=0.99,  # RL configurations
                 eps_start=1,  # exploration configurations
                 eps_end=0.01,
                 save_dir=None,  # saving configurations
                 model_name=None,
                 device='cpu',
                 ):
        self.device = torch.device(device)
        # environment
        self.env = env
        self.env_map = None
        self.maze_size = maze_list[0]
        self.maze_seed = seed_list[0]
        self.maze_size_list = maze_list
        self.maze_seed_list = seed_list
        self.maze_dist_list = dist_list
        self.fix_maze = fix_maze
        self.fix_start = fix_start
        self.fix_goal = fix_goal
        self.theme_list = ['MISHMASH']
        self.decal_list = [decal_freq]
        # agent
        self.agent = agent
        # state configurations
        self.use_true_state = use_true_state
        # goal-conditioned configurations
        self.use_goal = use_goal
        self.goal_dist = goal_dist
        self.valid_dist_list = list(range(1, goal_dist, 1))
        # training configurations
        self.sample_start_goal_num = sample_start_goal_num
        self.train_episode_num = train_episode_num
        self.start_train_step = start_train_step
        self.max_time_steps = max_time_steps
        self.max_steps_per_episode = episode_time_steps
        # replay buffer configurations
        if buffer_size:
            self.replay_buffer = memory.ReplayMemory(buffer_size, transition)
            self.TRANSITION = transition
        self.batch_size = batch_size
        self.use_replay_buffer = use_replay
        # HER configurations
        self.use_her = use_her
        self.her_future_k = future_k  # future strategy
        # optimization configurations
        self.learning_rate = learning_rate
        # rl related configuration
        self.gamma = gamma
        # exploration configuration
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.schedule = ml_schedule.LinearSchedule(eps_start, eps_end, max_time_steps / 2)
        # results statistics
        self.distance = []
        self.returns = []
        self.lengths = []
        self.policy_returns = []
        # saving settings
        self.model_name = model_name
        self.save_dir = save_dir

        # maximal distance
        self.max_dist = 2
        self.b2b_pdist = None

        # graph_buffer
        self.graph_buffer = []

    def train_local_goal_conditioned_dqn(self):
        """
        Function is used to train the locally goal-conditioned double DQN.
        """
        print("Experiment: Train a local goal-conditioned DQN.")
        # set the training statistics
        rewards = []
        episode_t = 0  # time step for one episode
        sample_start_goal_num = self.sample_start_goal_num
        train_episode_num = self.train_episode_num  # training number for each start-goal pair

        # initialize the state and goal
        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=self.fix_maze)

        # store the first state
        self.graph_buffer.append(state)

        # start the training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)

            # get action
            action = self.agent.get_action(state, goal, eps)

            # step in the environment
            next_state, reward, done, dist, trans, _, _ = self.env.step(action)
            episode_t += 1

            # store the replay buffer and convert the data to tensor
            if self.use_replay_buffer:
                # construct the transition
                trans = self.toTransition(state, action, next_state, reward, goal, done)
                # add the transition into the buffer
                self.replay_buffer.add(trans)

            state = next_state
            rewards.append(reward)

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                # compute the discounted return for each time step
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G

                # store the return, episode length, and final distance for current episode
                self.returns.append(G)
                self.lengths.append(episode_t)
                self.distance.append(dist)
                # compute the episode number
                episode_idx = len(self.returns)

                pbar.set_description(
                    f'Episode: {episode_idx} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f} | '
                    f'Init: {self.env.start_pos} | Goal: {self.env.goal_pos} | '
                    f'Eps: {eps:.3f} | Buffer: {len(self.replay_buffer)}'
                )

                # reset the environments
                rewards = []
                episode_t = 0
                # train a pair of start and goal with fixed number of episodes
                if sample_start_goal_num > 0:
                    if train_episode_num > 0:
                        # keep the same start and goal
                        self.fix_start = True
                        self.fix_goal = True
                        # add sampling here
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                        train_episode_num -= 1
                    else:
                        # sample a new pair of start and goal
                        self.fix_start = False
                        self.fix_goal = False
                        # constrain the distance <= max dist
                        self.fix_start = True
                        self.fix_goal = True
                        # self.goal_dist = random.sample(self.valid_dist_list, 1)[0]
                        self.goal_dist = 1
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                        train_episode_num = self.train_episode_num
                        sample_start_goal_num -= 1
                else:
                    # sample a new maze
                    self.fix_start = False
                    self.fix_goal = False
                    # constrain the distance <= max dist
                    # self.goal_dist = random.sample(self.valid_dist_list, 1)[0]
                    state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=True)
                    # reset the training control
                    train_episode_num = self.train_episode_num
                    sample_start_goal_num = self.sample_start_goal_num
                # store the first state
                self.graph_buffer.append(state)

            # train the agent
            if t > self.start_train_step:
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                self.agent.train_one_batch(t, sampled_batch)

        # save results
        self.save_results()

    def train_local_goal_conditioned_dqn_with_her(self):
        """
        Function is used to train the locally goal-conditioned double DQN.
        """
        # set the training statistics
        print("Experiment: Train a local goal-conditioned DQN with HER.")
        states = []
        actions = []
        rewards = []
        trans_poses = []
        dones = []
        episode_t = 0  # time step for one episode
        sample_start_goal_num = self.sample_start_goal_num  # sampled start and goal pair
        train_episode_num = self.train_episode_num  # training number for each start-goal pair

        # initialize the state and goal
        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=self.fix_maze)
        states.append(state)

        # start the training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)

            # get action
            action = self.agent.get_action(state, goal, eps)

            # step in the environment
            next_state, reward, done, dist, trans, _, _ = self.env.step(action)

            episode_t += 1
            # save the transitions
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            trans_poses.append(trans)
            dones.append(done)

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                # compute the discounted return for each time step
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G

                # store the return, episode length, and final distance for current episode
                self.returns.append(G)
                self.lengths.append(episode_t)
                self.distance.append(dist)
                # compute the episode number
                episode_idx = len(self.returns)

                # construct the memory buffer using HER
                self.hindsight_experience_replay(states, actions, rewards, trans_poses, goal, dones)

                pbar.set_description(
                    f'Episode: {episode_idx} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f} | '
                    f'Init: {self.env.start_pos} | Goal: {self.env.goal_pos} | '
                    f'Eps: {eps:.3f} | Buffer: {len(self.replay_buffer)}'
                )

                # reset the environments
                states = []
                actions = []
                trans_poses = []
                rewards = []
                dones = []
                episode_t = 0
                # train a pair of start and goal with fixed number of episodes
                if sample_start_goal_num > 0:
                    if train_episode_num > 0:
                        # keep the same start and goal
                        self.fix_start = True
                        self.fix_goal = True
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                        train_episode_num -= 1
                    else:
                        # sample a new pair of start and goal
                        self.fix_start = False
                        self.fix_goal = False
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                        train_episode_num = self.train_episode_num
                        sample_start_goal_num -= 1
                else:
                    # sample a new maze
                    self.fix_start = False
                    self.fix_goal = False
                    state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=True)
                    # reset
                    train_episode_num = self.train_episode_num
                    sample_start_goal_num = self.sample_start_goal_num
                states.append(state)
            else:
                state = next_state
                rewards.append(reward)

            # train the agent
            if t > self.start_train_step:
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                self.agent.train_one_batch(t, sampled_batch)

        # save results
        self.save_results()

    def test_distance_prediction(self):
        # load the saved data
        self.agent.policy_net.load_state_dict(torch.load('./sorb_test/test_5.pt'))
        self.agent.policy_net.eval()

        # sample a start-goal pair
        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=self.fix_maze)
        run_num = 10
        for r in range(run_num):
            gt_dist = len(self.env_map.path) - 1
            with torch.no_grad():
                state = self.toTensor(state)
                goal = self.toTensor(goal)
                pred_dist = self.agent.policy_net(state, goal)

            # for distributional RL
            pred_dist = torch.mm(pred_dist.squeeze(0), self.agent.support_atoms_values).max().item()
            # for normal DQN
            #pred_dist = np.round(pred_dist.max())

            print(f'State={state}, goal={goal}, GT={gt_dist} Pred={-1 * pred_dist}')
            self.fix_start = False
            self.fix_goal = False
            state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=self.fix_maze)

    def run_SoRB(self):
        # evaluation results
        eval_results = defaultdict()
        # load the policy
        self.agent.policy_net.load_state_dict(torch.load(f'/mnt/sda/rl_results/7x7/sorb_dqn_{self.maze_size}.pt'))
        self.agent.policy_net.eval()
        # load the replay buffer
        self.replay_buffer = np.load(f'./sorb_test/test_{self.maze_size}_buffer.npy')
        # init the environment
        self.update_map2d_and_maze3d(set_new_maze=True)
        # self.b2b_pdist = self.compute_pairwise_dist(mode='buffer-buffer')
        # np.save(f'./b2b_{self.maze_size}.npy', self.b2b_pdist)
        self.b2b_pdist = np.load(f'./b2b_{self.maze_size}.npy')
        total_pairs_dict = self.load_pair_data(self.maze_size, self.maze_seed)
        pairs_dict = {'start': None, 'goal': None}
        for g_dist in self.maze_dist_list:
            if not str(g_dist) in total_pairs_dict.keys():
                print(f"No pair with distance = {g_dist} is found.")
                break
            pairs_dict['start'] = total_pairs_dict[str(g_dist)][0]
            pairs_dict['goal'] = total_pairs_dict[str(g_dist)][1]
            run_num = 100
            success_count = 0
            #for r in range(run_num):
            #    idx = random.sample(range(len(pairs_dict['start'])), 1)[0]
            #    if random.uniform(0, 1) < 0.5:
            #        s_pos = pairs_dict['start'][idx]
            #        g_pos = pairs_dict['goal'][idx]
            #    else:
            #        s_pos = pairs_dict['goal'][idx]
            #        g_pos = pairs_dict['start'][idx]
            for s_pos, g_pos in zip(pairs_dict['start'], pairs_dict['goal']):
                #s_pos = [1, 3]
                #g_pos = [5, 1]
                state, goal, start_pos, goal_pos = self.update_maze_from_pos(s_pos, g_pos)
                print(f"Start pos = {start_pos}, Goal pos = {goal_pos}")
                # time steps
                max_time_steps = 20
                for t in range(max_time_steps):
                    # get an action
                    action, waypoint = self.search_policy(state, goal)
                    # take one step
                    next_state, reward, done, dist, trans, _, _ = self.env.step(action)
                    print(f"Step = {t}: state={state}, action={DEFAULT_ACTION_LIST[action]}, next_state={next_state}, waypoint={waypoint} done={done}")
                    # check terminal
                    if done:
                        success_count += 1
                        break
                    else:
                        state = next_state
                 
                # reverse direction
                tmp_pos = s_pos
                s_pos = g_pos
                g_pos = tmp_pos
                state, goal, start_pos, goal_pos = self.update_maze_from_pos(s_pos, g_pos)
                print(f"Start pos = {start_pos}, Goal pos = {goal_pos}")
                # time steps
                max_time_steps = 20
                for t in range(max_time_steps):
                    # get an action
                    action, waypoint = self.search_policy(state, goal)
                    # take one step
                    next_state, reward, done, dist, trans, _, _ = self.env.step(action)
                    print(f"Step = {t}: state={state}, action={DEFAULT_ACTION_LIST[action]}, next_state={next_state}, waypoint={waypoint} done={done}")
                    # check terminal
                    if done:
                        success_count += 1
                        break
                    else:
                        state = next_state
                
                print('-------------------------------')
            print(f"Success rate = {success_count / (len(pairs_dict['start']) * 2)}")
            eval_results[f"maze-{self.maze_size}-{self.maze_seed}-{g_dist}"] = success_count / (len(pairs_dict['start'] * 2))
        print("Evaluation finished")
        save_name = '/'.join([self.save_dir, f"eval_{self.maze_size}_policy.txt"])
        with open(save_name, "w") as f:
            for key, val in eval_results.items():
                tmp_str = key + " " + str(val) + '\n'
                f.write(tmp_str)
            f.close()

    def search_policy(self, state, goal):
        # compute the next way point
        # print('Search next way point')
        state_wp = self.shortest_path(state, goal)
        #print("Next way point = ", state_wp)
        # compute the distance between state and way point
        dist_s2wp = self.compute_distance(self.toTensor(state), self.toTensor(state_wp), true_dist=True)
        # compute the distance between state and goal
        dist_s2g = self.compute_distance(self.toTensor(state), self.toTensor(goal), true_dist=True)
        # compute the action to take
        #print(f"S2Way = {dist_s2wp}, S2Goal = {dist_s2g}")
        if dist_s2wp < dist_s2g or dist_s2g > self.max_dist:
            #print("Use way point")
            action = self.agent.get_action(self.toTensor(state), self.toTensor(state_wp), 0)
            waypoint = state_wp
        else:
            #print("Use raw goal")
            action = self.agent.get_action(self.toTensor(state), self.toTensor(goal), 0)
            waypoint = goal
        return action, waypoint

    def shortest_path(self, state, goal):
        # compute the distance matrices
        #print("compute buffer to buffer")
        pdist_b2b = self.b2b_pdist 
        pdist_s2b = self.compute_pairwise_dist(state=self.toTensor(state), mode='state-buffer')
        pdist_b2g = self.compute_pairwise_dist(goal=self.toTensor(goal), mode='buffer-goal')
        pdist_s2g = pdist_s2b + pdist_b2b + np.transpose(pdist_b2g)

        # find the index of the next way point
        min_index = np.argwhere(pdist_s2g == np.min(pdist_s2g))
        # get the next way point from the memory buffere
        # get the nearest waypoint
        waypoints_idx = []
        waypoints_dist = []
        for i in range(min_index.shape[0]):
            idx = min_index[i, 1]
            if pdist_s2b[0, idx] != 0:
                waypoints_idx.append(idx)
                waypoints_dist.append(pdist_s2b[0, idx])
        idx = waypoints_idx[waypoints_dist.index(np.min(waypoints_dist))] 
        next_way_point = self.replay_buffer[idx]
        return next_way_point

    def compute_pairwise_dist(self, state=None, goal=None, mode=None):
        state_num = self.replay_buffer.shape[0]
        # set different source tensor based on different mode
        if mode == 'state-buffer':
            pdist = np.zeros((1, state_num))
            for i in range(state_num):
                with torch.no_grad():
                    tmp_state = state
                    tmp_goal = torch.tensor(self.replay_buffer[i]).float()
                    dist = self.compute_distance(tmp_state, tmp_goal, true_dist=True)
                    pdist[0, i] = dist
            pdist = np.transpose(np.ones_like(pdist)) * pdist
        elif mode == 'buffer-goal':
            pdist = np.zeros((1, state_num))
            for i in range(state_num):
                with torch.no_grad():
                    tmp_state = goal
                    tmp_goal = torch.tensor(self.replay_buffer[i]).float()
                    dist = self.compute_distance(tmp_state, tmp_goal, true_dist=True)
                    pdist[0, i] = dist
            pdist = np.transpose(np.ones_like(pdist)) * pdist
        elif mode == 'buffer-buffer':
            # estimate the distance using the learned goal-conditioned value function
            pdist = torch.zeros((state_num, state_num))
            for i in range(state_num):
                for j in range(i+1, state_num):
                    with torch.no_grad():
                        tmp_state = torch.tensor(self.replay_buffer[i]).float()
                        tmp_goal = torch.tensor(self.replay_buffer[j]).float()
                        dist = self.compute_distance(tmp_state, tmp_goal, true_dist=False)
                        pdist[i, j] = dist
            pdist = pdist + pdist.t()
            pdist = scipy.sparse.csgraph.floyd_warshall(pdist, directed=True)
        else:
            raise Exception(f"Invalid mode input. Expected on in state-buffer, buffer-goal, or buffer-buffer,"
                            f"but get {mode}")
        return pdist

    def compute_distance(self, state, goal, true_dist=False):
        if not true_dist:
            with torch.no_grad():
                q_values = self.agent.policy_net(state, goal)
                # because our policy is a greedy policy
                v_value = q_values.max(dim=1)[0]
            return abs(v_value.item())
        else:
            state_map = [int(s) for s in state.numpy().tolist()][0:2]
            goal_map = [int(g) for g in goal.numpy().tolist()][0:2]
            
            dist = len(searchAlg.A_star(self.env_map.map2d_grid, state_map, goal_map)) - 1
            return dist

    def load_pair_data(self, m_size, m_seed):
       path = f"/mnt/sda/map/maze_{m_size}_{m_seed}.pkl"
       f = open(path, 'rb')
       return pickle.load(f)

    # adding transition to the buffer using HER
    def hindsight_experience_replay(self, states, actions, rewards, trans_poses, goal, dones):
        states_num = len(states) - 1
        for t in range(states_num):
            # extract one transition
            state = states[t]  # s_t
            action = actions[t]  # a_t
            reward = rewards[t]  # r_t
            next_state = states[t + 1]  # s_t+1
            next_state_pos = trans_poses[t]  # position of the current next state
            done = dones[t]  # done_t
            # normal replay buffer
            transition = self.toTransition(state, action, next_state, reward, goal,
                                           done)  # add the current transition
            self.replay_buffer.add(transition)
            # Hindsight Experience Replay
            future_indices = list(range(t + 1, len(states)))
            # sampled future goals
            sampled_goals = random.sample(future_indices, self.her_future_k) if len(
                future_indices) >= self.her_future_k else future_indices
            # relabel the current transition
            for idx in sampled_goals:
                new_goal = states[idx]
                new_goal_pos = trans_poses[idx - 1]
                distance = self.env.compute_distance(next_state_pos, new_goal_pos)
                new_reward = self.env.compute_reward(distance)
                new_done = 0 if new_reward == -1 else 1
                transition = self.toTransition(state, action, next_state, new_reward, new_goal, new_done)
                self.replay_buffer.add(transition)

    def toTransition(self, state, action, next_state, reward, goal, done):
        """
        Function is used to construct a transition using state, action, next_state, reward, goal, done.
        """
        if not self.use_goal:  # construct non goal-conditioned transition (Default type is int8 to save memory)
            return self.TRANSITION(state=self.toTensor(state),
                                   action=torch.tensor(action, dtype=torch.int8).view(-1, 1),
                                   reward=torch.tensor(reward, dtype=torch.int8).view(-1, 1),
                                   next_state=self.toTensor(next_state),
                                   done=torch.tensor(done, dtype=torch.int8).view(-1, 1))
        else:  # construct goal-conditioned transition
            return self.TRANSITION(state=self.toTensor(state),
                                   action=torch.tensor(action, dtype=torch.int8).view(-1, 1),
                                   reward=torch.tensor(reward, dtype=torch.int8).view(-1, 1),
                                   next_state=self.toTensor(next_state),
                                   goal=self.toTensor(goal),
                                   done=torch.tensor(done, dtype=torch.int8).view(-1, 1))

    def toTensor(self, obs_list):
        """
        Function is used to convert the data type. In the current settings, the state obtained from the environment is a
        list of 8 RGB observations (numpy arrays). This function will change the list into a tensor with size
        8 x 3 x 64 x 64.
        :param obs_list: List of the 8 observations
        :return: state tensor
        """
        if not self.use_true_state:  # convert the state observation from numpy to tensor with correct size
            state_obs = torch.tensor(np.array(obs_list).transpose(0, 3, 1, 2), dtype=torch.uint8)
        else:
            state_obs = torch.tensor(np.array(obs_list), dtype=torch.float32)
        return state_obs

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
            self.env_map.sample_random_start_goal_pos(self.fix_start, self.fix_goal, self.goal_dist)
            init_pos = self.env_map.init_pos
            goal_pos = self.env_map.goal_pos
            # initialize the maze 3D
            maze_configs["maze_name"] = f"maze_{self.maze_size}_{self.maze_seed}"  # string type name
            maze_configs["maze_size"] = [self.maze_size, self.maze_size]  # [int, int] list
            maze_configs["maze_seed"] = '1234'  # string type number
            maze_configs["maze_texture"] = random.sample(self.theme_list, 1)[0]  # string type name in theme_list
            maze_configs["maze_decal_freq"] = random.sample(self.decal_list, 1)[0]  # float number in decal_list
            maze_configs["maze_map_txt"] = "".join(self.env_map.map2d_txt)  # string type map
            maze_configs["maze_valid_pos"] = self.env_map.valid_pos  # list of valid positions
            # initialize the maze start and goal positions
            maze_configs["start_pos"] = self.env_map.init_pos + [
                0]  # start position on the txt map [rows, cols, orientation]
            maze_configs["goal_pos"] = self.env_map.goal_pos + [
                0]  # goal position on the txt map [rows, cols, orientation]
            # initialize the update flag
            maze_configs["update"] = True  # update flag
        else:
            init_pos, goal_pos = self.env_map.sample_random_start_goal_pos(self.fix_start, self.fix_goal, self.goal_dist)
            maze_configs['start_pos'] = init_pos + [0]
            maze_configs['goal_pos'] = goal_pos + [0]
            maze_configs['maze_valid_pos'] = self.env_map.valid_pos
            maze_configs['update'] = False

        # obtain the state and goal observation
        state_obs, goal_obs, _, _ = self.env.reset(maze_configs)
        # return states and goals
        return state_obs, goal_obs, init_pos, goal_pos
    
    def update_maze_from_pos(self, start_pos, goal_pos):
        maze_configs = defaultdict(lambda: None)
        self.env_map.update_mapper(start_pos, goal_pos) 
        maze_configs['start_pos'] = start_pos + [0]
        maze_configs['goal_pos'] = goal_pos + [0]
        maze_configs['maze_valid_pos'] = self.env_map.valid_pos
        maze_configs['maze_seed'] = '1234'
        maze_configs['update'] = False
        
        state_obs, goal_obs, _, _ = self.env.reset(maze_configs)
        
        return state_obs, goal_obs, start_pos, goal_pos

    # save the results
    def save_results(self):
        # compute the path for the results
        model_save_path = os.path.join(self.save_dir, self.model_name) + ".pt"
        # distance_save_path = os.path.join(self.save_dir, self.model_name + "_distance.npy")
        returns_save_path = os.path.join(self.save_dir, self.model_name + "_return.npy")
        # policy_returns_save_path = os.path.join(self.save_dir, self.model_name + "_policy_return.npy")
        # lengths_save_path = os.path.join(self.save_dir, self.model_name + "_length.npy")
        # save the memory buffer
        buffer_path = os.path.join(self.save_dir, self.model_name + "_buffer.npy")
        sampled_init_states = random.sample(self.graph_buffer, 1000)
        np.save(buffer_path, sampled_init_states)
        # save the results
        torch.save(self.agent.policy_net.state_dict(), model_save_path)
        # np.save(distance_save_path, self.distance)
        np.save(returns_save_path, self.returns)
        # np.save(lengths_save_path, self.lengths)
        # np.save(policy_returns_save_path, self.policy_returns)
