"""
    This is the script to register the customized environment to OpenAI gym
        - id: name of the environment, e.g. LabRandomMaze-v0
        - entry_point: package name.python script name:class name e.g. envs.LabEnv:RandomMaze
"""
from gym.envs.registration import register


register(
    id="LabRandomMaze-v0",
    entry_point="envs.LabEnv:RandomMaze"
)


