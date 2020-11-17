import numpy as np
from grid2op.Reward.BaseReward import BaseReward
from grid2op.dtypes import dt_float

class MyCloseToOverFlowReward(BaseReward):
    """
    This reward finds all lines close to overflowing.
    Returns max reward when there is no overflow, min reward if more than one line is close to overflow
    and the mean between max and min reward if one line is close to overflow
    """
    def __init__(self, max_lines=15):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)
        self.max_overflowed = dt_float(max_lines)

    def initialize(self, env):
        pass
        
    def __call__(self,  action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        thermal_limits = env.backend.get_thermal_limit()
        lineflow_ratio = env.current_obs.rho

        close_to_overflow = dt_float(0.0)
        for ratio, limit in zip(lineflow_ratio, thermal_limits):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= 0.95) or ratio >= 0.975:
                close_to_overflow += dt_float(1.0)
        
        #add by Jason
        # penalty on cascade failure.
        cooldown_line = env.current_obs.time_before_cooldown_line
        cascade_failure_list = np.where(cooldown_line == 12)[0]
        if len(cascade_failure_list) > 0:
            close_to_overflow += dt_float(len(cascade_failure_list)*0.5)  

        # penality weight on soft overflow
        close_to_overflow += 0.5 * sum((((env.current_obs.rho - 0.99)* \
                          (env.current_obs.timestep_overflow-1))[env.current_obs.timestep_overflow >= 2]))
        
        ## penalty on hard overflow.
        hard_overflow = np.where(env.current_obs.rho>2)[0]
        if len(hard_overflow) > 0:
            close_to_overflow += dt_float(len(hard_overflow)*0.5)  
        #added end

        
        close_to_overflow = np.clip(close_to_overflow,
                                    dt_float(0.0), self.max_overflowed)
        reward = np.interp(close_to_overflow,
                           [dt_float(0.0), self.max_overflowed],
                           [self.reward_max, self.reward_min])
        return reward


class MyPlusCloseToOverFlowReward(BaseReward):
    """
    This reward finds all lines close to overflowing.
    Returns max reward when there is no overflow, min reward if more than one line is close to overflow
    and the mean between max and min reward if one line is close to overflow
    """
    def __init__(self, max_lines=5):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)
        self.max_overflowed = dt_float(max_lines)
        self.power_rho = int(2)  # to which "power" is put the rho values

    def initialize(self, env):
        pass

    def __call__(self,  action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        thermal_limits = env.backend.get_thermal_limit()
        lineflow_ratio = env.current_obs.rho

        close_to_overflow = dt_float(0.0)
        for ratio, limit in zip(lineflow_ratio, thermal_limits):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= 0.95) or ratio >= 0.975:
                close_to_overflow += dt_float(1.0)

                obs = env.current_obs
                n_connected = np.sum(obs.line_status.astype(dt_float))
                #usage = np.sum(obs.rho[obs.line_status == True])
                usage = np.sum(obs.rho[obs.line_status == True]**self.power_rho)
                usage = np.clip(usage, 0.0, float(n_connected))
                reward = np.interp(n_connected - usage,
                                   [dt_float(0.0), float(n_connected)],
                                   [self.reward_min, self.reward_max])

                close_to_overflow += reward*dt_float(-0.1)



        close_to_overflow = np.clip(close_to_overflow,
                                    dt_float(0.0), self.max_overflowed)
        reward = np.interp(close_to_overflow,
                           [dt_float(0.0), self.max_overflowed],
                           [self.reward_max, self.reward_min])
        return reward



class MyLinesReconnectedReward(BaseReward):
    """
    This reward computes a penalty
    based on the number of off cooldown disconnected lines
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)
        self.penalty_max_at_n_lines = dt_float(2.0)


    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get obs from env
        obs = env.current_obs

        # All lines ids
        lines_id = np.arange(env.n_line)
        lines_id = lines_id[obs.time_before_cooldown_line == 0]

        n_penalties = dt_float(0.0)
        for line_id in lines_id:
            # Line could be reconnected but isn't
            if obs.line_status[line_id] == False:
                n_penalties += dt_float(1.0)

        max_p = self.penalty_max_at_n_lines
        n_penalties = np.clip(n_penalties, dt_float(0.0), max_p)
        r = np.interp(n_penalties,
                      [dt_float(0.0), max_p],
                      [self.reward_max, self.reward_min])
        return dt_float(r)


class MyLinesCapacityReward(BaseReward):
    """
    Reward based on lines capacity usage
    Returns max reward if no current is flowing in the lines
    Returns min reward if all lines are used at max capacity

    Compared to `:class:L2RPNReward`:
    This reward is linear (instead of quadratic) and only
    considers connected lines capacities
    """

    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)
        self.power_rho = int(2)  # to which "power" is put the rho values

    def initialize(self, env):
        pass

    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        obs = env.current_obs
        n_connected = np.sum(obs.line_status.astype(dt_float))
        """
        weight = np.ones(obs.rho.shape, dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        for i in np.where(thermal_limits < 400.00)[0]:
            weight[i]= weight[i]*1.05  #small line need reduced on weight 
        
        revised_rho = np.multiply(obs.rho, weight, dtype=dt_float)
        
        usage = np.sum(revised_rho[obs.line_status == True]**self.power_rho)
        """
        
        #usage = np.sum(obs.rho[obs.line_status == True])
        usage = np.sum(obs.rho[obs.line_status == True]**self.power_rho)
        
        usage = np.clip(usage, 0.0, float(n_connected))
        reward = np.interp(n_connected - usage,
                           [dt_float(0.0), float(n_connected)],
                           [self.reward_min, self.reward_max])
        return reward


class MySandboxReward(BaseReward):
    def __init__(self, alpha_redisph=1.0):
        BaseReward.__init__(self)
        self.reward_min = dt_float(-1.0)  # carefull here between min and max...
        self.reward_max = dt_float(1.0)
        self.alpha_redisph = dt_float(1.0)

    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        # added by Jason
        if has_error or is_illegal or is_ambiguous :
            return self.reward_min

        ######### scored reward ######### ######### #########
        # compute the losses
        gen_p, *_ = env.backend.generators_info()
        load_p, *_ = env.backend.loads_info()
        losses = np.sum(gen_p, dtype=dt_float) - np.sum(load_p, dtype=dt_float)

        # compute the marginal cost
        p_t = np.max(env.gen_cost_per_MW[env.gen_activeprod_t > 0.]).astype(dt_float)

        # redispatching amount
        c_redispatching = dt_float(2.0) * 1.0 * np.sum(np.abs(env.actual_dispatch)) * p_t

        # cost of losses
        c_loss = losses * p_t

        # total "operationnal cost"
        c_operations = dt_float(c_loss + c_redispatching)


        res = np.interp(c_operations,
                   [1, 300.0 * 70.0],
                   [1, -1])



        return dt_float(res)

class MyDistanceReward(BaseReward):
    """
    This reward computes a penalty based on the distance of the current grid to the grid at time 0.
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get topo from env
        obs = env.current_obs
        topo = obs.topo_vect

        idx = 0
        diff = dt_float(0.0)
        for n_elems_on_sub in obs.sub_info:
            # Find this substation elements range in topology vect
            sub_start = idx
            sub_end = idx + n_elems_on_sub
            current_sub_topo = topo[sub_start:sub_end]

            # Count number of elements not on bus 1
            # Because at the initial state, all elements are on bus 1
            diff += dt_float(1.0) * np.count_nonzero(current_sub_topo != 1)

            # Set index to next sub station
            idx += n_elems_on_sub

        r = np.interp(diff,
                      [dt_float(0.0), len(topo) * dt_float(1.0)],
                      [self.reward_max, self.reward_min])
        return r

class MyL2RPNReward(BaseReward):
    def __init__(self):
        BaseReward.__init__(self)

    def initialize(self, env):
        self.reward_min = dt_float(0.0)
        #self.reward_max = dt_float(env.backend.n_line)  #changed by Jason
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            line_cap = self.__get_lines_capacity_usage(env)
            #res = np.sum(line_cap)
            res = np.sum(line_cap[env.current_obs.line_status == True])
            # added by Jason
            res /= env.n_line
            if not np.isfinite(res):
                res = self.reward_min
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        return res

    @staticmethod
    def __get_lines_capacity_usage(env):
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        thermal_limits += 1e-1  # for numerical stability
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)

        x = np.minimum(relative_flow, dt_float(1.0))
        lines_capacity_usage_score = np.maximum(dt_float(1.0) - x ** 2, np.zeros(x.shape, dtype=dt_float))
        for i in np.where(thermal_limits < 400.00)[0]:
            lines_capacity_usage_score[i]= lines_capacity_usage_score[i]*0.7  #small line need reduced on weight 
        return lines_capacity_usage_score

        
class MyNewReward(BaseReward):
    def __init__(self):
        super().__init__()
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)

    def initialize(self, env):
        pass

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            line_cap = self.__get_lines_capacity_usage(env)
            state_obs = env.current_obs
            
            ## changed by Jason to remove the capacity of disconnected lines
            #cap = np.sum(line_cap)
            cap = np.sum(line_cap[state_obs.line_status == True])
            
            res = cap - 150 * sum((((state_obs.rho - 0.99))[state_obs.rho > 1]))
            # penality weight on soft overflow
            res = res - 15 * sum((((state_obs.rho - 0.99)* \
                              (state_obs.timestep_overflow-1))[state_obs.timestep_overflow >= 2]))
            
            ## penalty on hard overflow.
            hard_overflow = np.where(state_obs.rho>2)[0]
            if len(hard_overflow) > 0:
                res = res - dt_float(len(hard_overflow)*15)  
                        
            ## penalty on cascade failure.
            cooldown_line = state_obs.time_before_cooldown_line
            cascade_failure_list = np.where(cooldown_line == 12)[0]
            if len(cascade_failure_list) > 0:
                res = res - dt_float(len(cascade_failure_list)*15)  
            
            if res < -30.0:
                res = -30.0
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = -30.0
        res = np.interp(res,
                      [dt_float(-30.0), 186.0],
                      [self.reward_min, self.reward_max])
        return res

    def __get_lines_capacity_usage(self,env):
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        thermal_limits += 1e-1  # for numerical stability
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)
        x = np.minimum(relative_flow, dt_float(1.0))
        lines_capacity_usage_score = np.maximum(dt_float(1.0) - x ** 2, np.zeros(x.shape, dtype=dt_float))
        return lines_capacity_usage_score        


class MyLinesConnectedReward(BaseReward):
    """
    This reward computes a penalty
    based on the number of off cooldown disconnected lines
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)
        self.penalty_max_at_n_lines = dt_float(10.0)

    def __call__(self, action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            # previous action was bad
            res = self.reward_min
        elif is_done:
            # really strong reward if an episode is over without game over
            res = self.reward_max
        else:
            # Get obs from env
            obs = env.current_obs

            # All lines ids
            lines_id = np.arange(env.n_line)
            #lines_id = lines_id[obs.time_before_cooldown_line != 0]

            n_penalties = dt_float(0.0)
            for line_id in lines_id:
                # Line is disconnected
                if obs.line_status[line_id] == False:
                    # check the cooldown duration
                    if obs.time_before_cooldown_line[line_id] <= 3: 
                        n_penalties += dt_float(0.3)  # it is disconnected by us. default: 3, less penalty
                    else:
                        n_penalties += dt_float(1.0)  # attacked or it is disconnected by system. default: 6
                    
            max_p = self.penalty_max_at_n_lines
            n_penalties = np.clip(n_penalties, dt_float(0.0), max_p)
            res = np.interp(n_penalties,
                          [dt_float(0.0), max_p],
                          [self.reward_max, self.reward_min])
        return dt_float(res)
        
class MyBalanceReward(BaseReward):
    """
    Reward based on lines capacity usage
    Returns max reward if no current is flowing in the lines
    Returns min reward if all lines are used at max capacity

    Compared to `:class:L2RPNReward`:
    This reward is linear (instead of quadratic) and only
    considers connected lines capacities
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)

    def initialize(self, env):
        pass

    def __call__(self,  action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        obs = env.current_obs
        if all(obs.line_status == True):
            return self.reward_max
        else:
            obs = env.current_obs
            line_rhos = obs.rho[obs.line_status == True]
            std = np.std(line_rhos)
            res = 1 - std
            if res < 0:
                res = 0

        return res
        