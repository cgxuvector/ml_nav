from model.GoalDQN import DQN
from utils import ml_schedule
from utils import mapper
import numpy as np
from collections import namedtuple
import tqdm
from utils import memory
import matplotlib.pyplot as plt

DEFAULT_TRANSITION = namedtuple("transition", ["state", "action", "reward", "next_state", "goal", "done"])


class Experiment(object):
    def __init__(self,
                 env,
                 maze_list,
                 seed_list,
                 agent,
                 max_time_steps,
                 max_time_steps_per_episode=100,
                 start_train_step=100,
                 buffer_size=None,
                 transition=DEFAULT_TRANSITION,
                 learning_rate=1e-3,
                 batch_size=64,
                 gamma=0.99,
                 save_dir=None,
                 model_name=None,
                 random_seed=1234,
                 use_replay=False):
        # environment
        self.env = env
        self.maze_list = maze_list
        self.seed_list = seed_list
        # agent
        self.agent = agent
        # training configurations
        self.max_time_steps = max_time_steps
        self.max_steps_per_episode = max_time_steps_per_episode
        self.start_train_step = start_train_step
        self.learning_rate = learning_rate
        # replay buffer configurations
        if buffer_size:
            self.replay_buffer = memory.ReplayMemory(buffer_size, transition)
            self.TRANSITION = transition
        self.batch_size = batch_size
        self.use_relay_buffer = use_replay
        # rl related configuration
        self.gamma = gamma
        # results statistics
        self.returns = []
        self.lengths = []
        # saving settings
        self.model_name = model_name
        self.save_dir = save_dir
        # random setting
        self.seed_rnd = random_seed

    def run(self):
        # running statistics
        rewards = []
        episode_t = 0
        episode = 0
        # reset environment
        np.random.seed(self.seed_rnd)
        size = np.random.choice(self.maze_list)
        seed = np.random.choice(self.seed_list)
        env_map = mapper.RoughMap(size, seed, 3)
        pos_params = env_map.get_start_goal_pos()
        state, goal = self.env.reset(size, seed, pos_params)
        pbar = tqdm.trange(self.max_time_steps)
        fig, arr = plt.subplots(1)
        img_art = arr.imshow(self.env.top_down_obs)
        for t in pbar:
            # get action
            action = self.agent.get_action(state, goal)
            # apply the action
            next_state, reward, done, _ = self.env.step(action)
            img_art.set_data(self.env.top_down_obs)
            fig.canvas.draw()
            plt.pause(0.001)
            # store the replay buffer
            if self.use_relay_buffer:
                trans = self.TRANSITION(state=None,
                                        action=None,
                                        reward=None,
                                        next_state=None,
                                        goal=None,
                                        done=None)
                self.replay_buffer.add(trans)

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                # compute the return
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G
                self.returns.append(G)
                episode = len(self.returns)
                self.lengths.append(len(rewards))
                pbar.set_description(
                    f'Episode: {episode} | Steps: {episode_t} | Return: {G}'
                )
                # reset
                rewards = []
                episode_t = 0
                size, seed, pos_params = self.map_sampling(env_map, self.maze_list, self.seed_list, False)
                state, goal = self.env.reset(size, seed, pos_params)
            else:
                state = next_state
                rewards.append(reward)
                episode_t += 1

    @staticmethod
    def map_sampling(env_map, maze_list, seed_list, sample_pos=False):
        if not sample_pos:
            size = np.random.choice(maze_list)
            seed = np.random.choice(seed_list)
            env_map = mapper.RoughMap(size, seed, 3)
            # positions
            pos_params = [env_map.raw_pos['init'][0],
                          env_map.raw_pos['init'][1],
                          env_map.raw_pos['goal'][0],
                          env_map.raw_pos['goal'][1],
                          0]  # [init_pos, goal_pos, init_orientation]
        else:
            size = env_map.maze_size
            seed = env_map.maze_seed
            start_idx, end_idx = np.random.choice(len(env_map.valid_pos), 2).tolist()
            start_pos = env_map.valid_pos[start_idx]
            end_pos = env_map.valid_pos[end_idx]
            # positions
            pos_params = [start_pos[0],
                          start_pos[1],
                          end_pos[0],
                          end_pos[1],
                          0]  # [init_pos, goal_pos, init_orientation]

        return size, seed, pos_params


