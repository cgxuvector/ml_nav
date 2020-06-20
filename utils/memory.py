from collections import namedtuple
from collections import deque
import numpy as np
import torch
import IPython.terminal.debugger as Debug

DEFAULT_TRANSITION = namedtuple("transition", ["state", "action", "reward", "next_state", "done"])


class ReplayMemory(object):
    """
        Define the experience replay buffer
        Note: currently, the replay store numpy arrays
    """

    def __init__(self, max_memory_size, transition=DEFAULT_TRANSITION):
        """
        Initialization function
        :param max_memory_size: maximal size of the replay memory
        :param transition: transition type defined as a named tuple. Default version is
                    "
                    Transition(state, action, next_state, reward, done)
                    "
        """
        # memory params
        self.max_size = max_memory_size
        self.size = 0

        # transition params
        self.TRANSITION = transition

        # memory data
        self.data_buffer = deque(maxlen=self.max_size)

    def __len__(self):
        """ Return current memory size. """
        return self.size

    def add(self, single_transition):
        """
        Add one transition to the replay memory
        :param trans: should be an instance of the namedtuple defined in self.TRANSITION
        :return: None
        """
        # check the fields compatibility
        assert (single_transition._fields == self.TRANSITION._fields), f"A valid transition should contain " \
                                                                       f"{self.TRANSITION._fields}" \
                                                                       f" but currently got {single_transition._fields}"
        # add the data into buffer
        self.data_buffer.append(single_transition)

        # track the current buffer size and index
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample one batch from the replay memory
        :param batch_size: size of the sampled batch
        :return: batch
        """
        assert batch_size > 0, f"Invalid batch size. Expect batch size > 0, but get {batch_size}"
        # sample the indices: if buffer is smaller than batch size,then return the whole buffer,
        # otherwise return the batch
        sampled_indices = np.random.choice(self.size, min(self.size, batch_size))
        # obtain the list of named tuples, each is a transition
        sampled_transition_list = [self.data_buffer[idx] for idx in sampled_indices]
        # convert the list of transitions to transition of list-like data
        # *sampled_list: unpack the list into elements
        # zip(*sampled_list): parallel iterate the sub-elements in each unpacked element
        # trans_type(*zip(*sample_list)): construct the batch
        sampled_transitions = self.TRANSITION(*zip(*sampled_transition_list))
        return sampled_transitions

    def save_buffer(self):
        state_shape = tuple([self.size] + list(self.data_buffer[0].state.size()))
        action_shape = tuple([self.size] + list(self.data_buffer[0].action.size()))
        reward_shape = tuple([self.size] + list(self.data_buffer[0].reward.size()))
        done_shape = tuple([self.size] + list(self.data_buffer[0].done.size()))

        state_buffer = torch.empty(state_shape)
        next_state_buffer = torch.empty(state_shape)
        goal_buffer = torch.empty(state_shape)
        action_buffer = torch.empty(action_shape)
        reward_buffer = torch.empty(reward_shape)
        done_buffer = torch.empty(done_shape)

        for idx, trans in enumerate(self.data_buffer):
            state_buffer[idx] = trans.state
            action_buffer[idx] = trans.action
            reward_buffer[idx] = trans.reward
            next_state_buffer[idx] = trans.next_state
            done_buffer[idx] = trans.done
            goal_buffer[idx] = trans.goal

        torch.save(state_buffer, './results/buffer/state_buffer.pt')
        torch.save(action_buffer, './results/buffer/action_buffer.pt')
        torch.save(reward_buffer, './results/buffer/reward_buffer.pt')
        torch.save(next_state_buffer, './results/buffer/next_state_buffer.pt')
        torch.save(done_buffer, './results/buffer/done_buffer.pt')
        torch.save(goal_buffer, './results/buffer/goal_buffer.pt')




