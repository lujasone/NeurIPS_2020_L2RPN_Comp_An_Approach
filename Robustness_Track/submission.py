from .agent_exp import MyAgent
from .MyCombinedScaledReward import *
    
def make_agent(env, submission_dir):
    """
    This function will be used by codalab to create your agent. It should accept exactly an environment and a path
    to your submission directory and return a valid agent.
    """
    agent = MyAgent(env.observation_space,
                       env.action_space)
    import os
    agent.load(os.path.join(submission_dir, "models", "DDDQN.h5"))
    return agent
    
# reward must be a subclass of grid2op.Reward.BaseReward.BaseReward:
reward = MyCombinedScaledReward # you can also create your own reward class



