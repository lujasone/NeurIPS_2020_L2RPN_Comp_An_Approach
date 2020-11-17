from .agent_exp import MyAgent
from grid2op.Reward import ConstantReward
from .MyRewards import *
from .MyCombinedScaledReward import *
import os
def make_agent(env, submission_dir):
    """
    This function will be used by codalab to create your agent. It should accept exactly an environment and a path
    to your submission directory and return a valid agent.
    """
    agent = MyAgent(env.observation_space,
                      env.action_space)
    agent.load(os.path.join(submission_dir, "models", "DDQN_XK.h5"))
    return agent

# reward must be a subclass of grid2op.Reward.BaseReward.BaseReward:
#reward = MySandboxReward
reward = MyCombinedScaledReward # you can also create your own reward class


