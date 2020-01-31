import six
import numpy as np
import random
import pprint
import cv2


# generate action tuple
def _action(*entries):
    return np.array(entries, dtype=np.intc)


class NavAgent(object):
    """Navigation agent in DeepMind Lab"""

    # define the actions
    """ Build-int Actions:  must be integers (discrete actions)
        LOOK_LEFT_RIGHT_PIXELS_PER_FRAME	[-512, 512]	   Look left/right angular (related)
        LOOK_DOWN_UP_PIXELS_PER_FRAME	    [-512, 512]	   Look down/up angular	
        STRAFE_LEFT_RIGHT	                [-1, 1]	       Strafe left/right.
        MOVE_BACK_FORWARD	                [-1, 1]	       Move back/forward (related)
        FIRE	                            [0, 1]	       Fire button held.
        JUMP	                            [0, 1]	       Jump button held.
        CROUCH
    """
    ACTIONS = {
        'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
        'look_right': _action(20, 0, 0, 0, 0, 0, 0),
        # 'straft_left': _action(0, 0, -1, 0, 0, 0, 0),
        # 'straft_right': _action(0, 0, 1, 0, 0, 0, 0),
        'forward': _action(0, 0, 0, -1, 0, 0, 0),
        'backward': _action(0, 0, 0, 1, 0, 0, 0)
        # 'look_left': _action(256, 0, 0, 1, 0, 0, 0),
        # 'look_right': _action(0, 0, 0, 0, 0, 0, 0),
        # 'straft_left': _action(0, 0, 0, 0, 0, 0, 0),
        # 'straft_right': _action(0, 0, 0, 0, 0, 0, 0),
        # 'forward': _action(0, 0, 0, 0, 0, 0, 0),
        # 'backward': _action(0, 0, 0, 0, 0, 0, 0)
    }

    # obtain the values of the 4 actions in a safe way
    # six.viewvalues() : py2.viewvalues() and py3.values()
    # for the dictionray variables
    # convert the legal actions to a list, each element
    # is still a 7 dimensional array
    ACTION_LIST = list(six.viewvalues(ACTIONS))
    ACTION_NAME_LIST = list(six.viewkeys(ACTIONS))

    def __init__(self, action_spec):
        self.action_spec = action_spec  # Dict of whole action space
        self.action_vals = NavAgent.ACTION_LIST  # selected actions values
        self.action_names = NavAgent.ACTION_NAME_LIST  # selected action names
        self.action_count = len(NavAgent.ACTION_LIST)  # number of used actions

    # print info
    def print_info(self):
        pprint.pprint(self.action_spec)
        for name, val in zip(self.action_names, self.action_vals):
            pprint.pprint("{} - {}".format(name, val))
        print("Action number: {}".format(self.action_count))

    # return an random action since no observation and state is given
    def step(self, obs):
        return self.action_vals[np.random.randint(0, len(self.action_vals))]


