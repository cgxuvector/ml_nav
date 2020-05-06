import gym

# test code
if __name__ == '__main__':
    my_env = gym.make('MountainCar-v0')
    my_env.reset()

    for i in range(10000):
        my_env.render()
        action = my_env.action_space.sample()
        next_state, reward, done, _ = my_env.step(action)
        if done:
            my_env.reset()

    my_env.close()
