class RandomAgent(object):
    def __init__(self, action_space, random_seed):
        self.action_space = action_space
        self.seed_rnd = random_seed

    def get_action(self, state, goal):
        return self.action_space.sample()
