import numpy as np

from grid2op.Reward.CombinedReward import CombinedReward
from grid2op.dtypes import dt_float

from grid2op.Reward import *
from .MyRewards import *

class MyCombinedScaledReward(CombinedReward):
    """
    This class allows to combine multiple rewards.
    It will compute a scaled reward of the weighted sum of the registered rewards.
    Scaling is done by linearly interpolating the weighted sum,
    from the range [min_sum; max_sum] to [reward_min; reward_max]

    min_sum and max_sum are computed from the weights and ranges of registered rewards.
    See `Reward.BaseReward` for setting the output range.
    """

    def __init__(self):
        super().__init__()
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)
        self._sum_max = dt_float(0.0)
        self._sum_min = dt_float(0.0)
        self.rewards = {}


    def initialize(self, env):
        """
        Overloaded initialze from `Reward.CombinedReward`.
        This is because it needs to store the ranges internaly
        """
        self._sum_max = dt_float(0.0)
        self._sum_min = dt_float(0.0)

        # added by Jason
        self.addReward("sbox", MySandboxReward(), 30.0)
        self.addReward("overflow", MyCloseToOverFlowReward(), 200.0)
        self.addReward("dist", MyDistanceReward(), 20.0)
        #self.addReward("cap", MyLinesCapacityReward(), 3.0)
        self.addReward("cap", MyNewReward(), 3.0)

        for key, reward in self.rewards.items():
            reward_w = dt_float(reward["weight"])
            reward_instance = reward["instance"]
            reward_instance.initialize(env)
            self._sum_max += dt_float(reward_instance.reward_max * reward_w)
            self._sum_min += dt_float(reward_instance.reward_min * reward_w)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        # Get weighted sum from parent
        ws = super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
        # Scale to range
        res = np.interp(ws, [self._sum_min, self._sum_max], [self.reward_min, self.reward_max])
        return dt_float(res)
