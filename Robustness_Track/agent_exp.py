from grid2op.Agent import BaseAgent
from .DoubleDuelingDQN import DoubleDuelingDQN as BackupAgent
import numpy as np
from grid2op.dtypes import dt_int, dt_float, dt_bool
import itertools
import math
import copy

MAX_BUS_CHANGE_SET = 5
HUB_SUBs = [16]
COMPLEX_SUBs = [26, 4]


class MyAgent(BaseAgent):
    def __init__(self,observation_space, action_space):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)
        self.obs_space = observation_space
        self.backupAgent = BackupAgent(observation_space, action_space,name = "DDDQN")
        self.lines_in_attack = []
        self.subs_in_attack = []
        self.agent_count = 0

    def load(self, path):
        self.backupAgent.Qmain.load_network(path)

    def change_bus_actions_by_id(self, action_space, sub_num, observation):
        action_array = []
        topo_objs = []
        sub_objs = action_space.get_obj_connect_to(substation_id=sub_num)
        topo_list = observation.state_of(substation_id=sub_num)["topo_vect"]
        or_ids = sub_objs["lines_or_id"]
        ex_ids = sub_objs["lines_ex_id"]
        if len(or_ids) + len(ex_ids) < 3:
            return action_array
        for or_id in or_ids:
            topo_pos = action_space.line_or_to_sub_pos[or_id]
            if or_id == 2 or or_id == 19 or or_id == 28 or or_id == 38 or or_id == 49:
                continue
            if topo_list[topo_pos] == -1:
                continue
            else:
                topo_obj = "or_" + str(or_id)
                topo_objs.append(topo_obj)

        for ex_id in ex_ids:
            topo_pos = action_space.line_ex_to_sub_pos[ex_id]
            if ex_id == 2 or ex_id == 19 or ex_id == 28 or ex_id == 38 or ex_id == 49:
                continue
            if topo_list[topo_pos] == -1:
                continue
            else:
                topo_obj = "ex_" + str(ex_id)
                topo_objs.append(topo_obj)

        for i in range(1, len(topo_objs) + 1):
            iter = itertools.combinations(topo_objs, i)
            iter = list(iter)
            for j in range(len(iter)):
                action_list = list(iter[j])
                or_line = []
                ex_line = []
                # added by Jason
                if sub_num in HUB_SUBs:
                    max_bus_change_set = 8
                elif sub_num in COMPLEX_SUBs:
                    max_bus_change_set = 6
                else:
                    max_bus_change_set = MAX_BUS_CHANGE_SET

                if len(action_list) <= max_bus_change_set and len(action_list) > 1:
                    for k in range(len(action_list)):
                        if "or_" in action_list[k]:
                            or_line.append(int(action_list[k][3:]))
                            if int(action_list[k][3:]) == 18 or int(action_list[k][3:]) == 27 or \
                                    int(action_list[k][3:]) == 37 or int(action_list[k][3:]) == 48:
                                or_line.append(int(action_list[k][3:]) + 1)
                        if "ex_" in action_list[k]:
                            ex_line.append(int(action_list[k][3:]))
                            if int(action_list[k][3:]) == 18 or int(action_list[k][3:]) == 27 or \
                                    int(action_list[k][3:]) == 37 or int(action_list[k][3:]) == 48:
                                ex_line.append(int(action_list[k][3:]) + 1)
                    action = action_space(
                        {"change_bus": {"lines_or_id": or_line, "lines_ex_id": ex_line}})
                    action_array.append(action)
        return action_array

    def change_bus_actions_by_id_old(self, action_space, sub_num, observation):
        action_array = []

        topo_objs = []
        sub_objs = action_space.get_obj_connect_to(substation_id=sub_num)
        topo_list = observation.state_of(substation_id=sub_num)["topo_vect"]
        or_ids = sub_objs["lines_or_id"]
        ex_ids = sub_objs["lines_ex_id"]
        for or_id in or_ids:
            topo_pos = action_space.line_or_to_sub_pos[or_id]
            if or_id == 2 or or_id == 19 or or_id == 28 or or_id == 38 or or_id == 49:
                continue
            if topo_list[topo_pos] == -1:
                continue
            else:
                topo_obj = "or_" + str(or_id)
                topo_objs.append(topo_obj)

        for ex_id in ex_ids:
            topo_pos = action_space.line_ex_to_sub_pos[ex_id]
            if ex_id == 2 or ex_id == 19 or ex_id == 28 or ex_id == 38 or ex_id == 49:
                continue
            if topo_list[topo_pos] == -1:
                continue
            else:
                topo_obj = "ex_" + str(ex_id)
                topo_objs.append(topo_obj)

        for i in range(1, len(topo_objs) + 1):
            iter = itertools.combinations(topo_objs, i)
            iter = list(iter)
            for j in range(len(iter)):
                action_list = list(iter[j])
                or_line = []
                ex_line = []
                if len(action_list) <= MAX_BUS_CHANGE_SET:
                    for k in range(len(action_list)):
                        if "or_" in action_list[k]:
                            or_line.append(int(action_list[k][3:]))
                            if int(action_list[k][3:]) == 18 or int(action_list[k][3:]) == 27 or \
                                    int(action_list[k][3:]) == 37 or int(action_list[k][3:]) == 48:
                                or_line.append(int(action_list[k][3:]) + 1)
                        if "ex_" in action_list[k]:
                            ex_line.append(int(action_list[k][3:]))
                            if int(action_list[k][3:]) == 18 or int(action_list[k][3:]) == 27 or \
                                    int(action_list[k][3:]) == 37 or int(action_list[k][3:]) == 48:
                                ex_line.append(int(action_list[k][3:]) + 1)
                    action = action_space({"change_bus": {"lines_or_id": or_line, "lines_ex_id": ex_line}})
                    action_array.append(action)
        return action_array

    def check_cooldown_legal(self, observation, action):
        lines_impacted, subs_impacted = action.get_topological_impact()
        line_need_cooldown = lines_impacted & observation.time_before_cooldown_line
        if np.any(line_need_cooldown):
            return False
        sub_need_cooldown = subs_impacted & observation.time_before_cooldown_sub
        if np.any(sub_need_cooldown):
            return False
        return True

    def choose_best_action_by_simul(self, observation, tested_action):
        # choose best action based on simulated reward
        best_action = None
        highest_reward = None
        if len(tested_action) > 1:
            resulting_rewards = np.full(shape=len(tested_action), fill_value=np.NaN, dtype=dt_float)
            for i, action in enumerate(tested_action):
                if self.check_cooldown_legal(observation, action) == False:
                    continue
                simul_obs, simul_reward, simul_has_error, simul_info = observation.simulate(action)
                resulting_rewards[i] = simul_reward
            reward_idx = int(np.nanargmax(resulting_rewards))  # rewards.index(max(rewards))
            highest_reward = np.max(resulting_rewards)
            best_action = tested_action[reward_idx]
        # only one action to be done
        elif len(tested_action) == 1:
            best_action = tested_action[0]
            simul_obs, highest_reward, simul_has_error, simul_info = observation.simulate(best_action)

        return best_action, highest_reward

    def add_to_attack_sub_list(self, chosen_action):
        lines_impacted, subs_impacted = chosen_action.get_topological_impact()
        change_subs = np.where(subs_impacted == True)[0]
        if len(change_subs) > 0:
            self.subs_in_attack.append(change_subs[0])
            self.subs_in_attack = list(set(self.subs_in_attack))

    def remove_from_attack_sub_list(self, chosen_action):
        lines_impacted, subs_impacted = chosen_action.get_topological_impact()
        change_subs = np.where(subs_impacted == True)[0]
        if len(change_subs) > 0:
            if change_subs[0] in self.subs_in_attack:
                self.subs_in_attack.remove(change_subs[0])

    def add_to_attack_line_list(self, observation):
        cooldown = observation.time_before_cooldown_line
        attack_lines = np.where(cooldown >= 13)[0]
        if len(attack_lines) > 0:
            self.lines_in_attack.extend(attack_lines)
            self.lines_in_attack = list(set(self.lines_in_attack))

    def remove_from_attack_line_list(self, action):
        lines_impacted, subs_impacted = action.get_topological_impact()
        change_lines = np.where(lines_impacted == True)[0]
        if len(change_lines) > 0:
            if change_lines[0] in self.lines_in_attack:
                self.lines_in_attack.remove(change_lines[0])

    def compare_with_model(self,act,rwd):
        if rwd > 0:
            return act
        chosen_action,chosen_rwd = self.backupAgent.get_action_by_model()
        if chosen_action is not None and chosen_rwd > rwd:
            self.agent_count += 1
            return chosen_action
        return act

    def act(self, observation, reward, done):
        state = self.backupAgent.convert_obs(observation)
        self.backupAgent._save_current_frame(state)
        self.backupAgent.obs = observation
        self.obs = observation

        self.add_to_attack_line_list(observation)

        ##### LineReconnection ########
        # we try to reconnect a line if possible, THIS IS TOP ACTION.
        action_reco = self.reco_line(observation)
        if action_reco is not None:
            self.remove_from_attack_line_list(action_reco)
            return action_reco

        ###### Other cases.#######################
        line_stat_s = copy.deepcopy(observation.line_status)
        cooldown = copy.deepcopy(observation.time_before_cooldown_line)
        rho = copy.deepcopy(observation.rho)

        attacked = np.any(~line_stat_s & (cooldown >= 13))
        disconnected = np.any(~line_stat_s & (cooldown < 13) & (cooldown > 0))
        all_connected = np.all(line_stat_s)
        overflowed = np.any(rho >= 1.0)
        no_overflowed = np.all(rho < 1.0)

        # case1 : attacked & overflowed
        if attacked and overflowed:
            # for all substations we get all possible topology action list
            # select topo actions.
            action_topo,rwd_topo = self.change_bus_topology_dispatch_emergency_disc(observation)  # from all subs.
            chosen_action = self.compare_with_model(action_topo, rwd_topo)
            if chosen_action is not None:
                self.add_to_attack_sub_list(chosen_action)
                return chosen_action
            return self.action_space({})  # DoNothing

        # case 2: attacked & No overflow
        elif attacked and no_overflowed:
            return self.action_space({})  # DoNothing

        # case3: dicconnected (from attacked or system-discon) and overflowed
        elif disconnected and overflowed:
            # same as attacked and overflowed
            action_topo,rwd_topo = self.change_bus_topology_dispatch_emergency_disc(observation)  # from all subs.
            chosen_action = self.compare_with_model(action_topo, rwd_topo)
            if chosen_action is not None:
                self.add_to_attack_sub_list(chosen_action)
                return chosen_action
            return self.action_space({})  # DoNothing

        # case 4: disconnected and no overflowed
        elif disconnected and no_overflowed:
            # recover topo due to attack.
            action_recover,rwd_recover = self.recover_reference_topology(observation)
            if action_recover is not None:
                self.remove_from_attack_sub_list(action_recover)
                return action_recover
            return self.action_space({})  # DoNothing

        # case 5: all-connected and overflowed
        elif all_connected and overflowed:
            # disconnect lines ourselves.
            action_discon,rwd_discon = self.try_out_disconnections(observation)
            # recover topo due to attack.
            action_recover, rwd_recover = self.recover_reference_topology(observation)
            if action_discon is None or action_recover is None:
                if action_discon is None and action_recover is not None:
                    return action_recover
                elif action_discon is not None and action_recover is None:
                    return action_discon
                return self.action_space({})
            # select the best action from 2 candidated actions
            elif rwd_discon >= rwd_recover:
                chosen_action = action_discon
                return chosen_action
            else:
                chosen_action = action_recover
                self.remove_from_attack_sub_list(action_recover)
                return chosen_action

        # case 6: all-connected and no_overflowed
        elif all_connected and no_overflowed:
            # recover topo due to attack.
            action_recover, rwd_recover = self.recover_reference_topology(observation)
            if action_recover is not None:
                self.remove_from_attack_sub_list(action_recover)
                return action_recover
            return self.action_space({})  # DoNothing

        else:
            return self.action_space({})  # DoNothing

    def reco_line(self, observation):
        line_stat_s = copy.deepcopy(observation.line_status)
        cooldown = copy.deepcopy(observation.time_before_cooldown_line)
        can_be_reco = ~line_stat_s & (cooldown == 0)
        if np.any(can_be_reco):
            res = []
            for id_ in np.where(can_be_reco)[0]:
                change_status = self.action_space.get_change_line_status_vect()
                change_status[id_] = True
                action = self.action_space({"change_line_status": change_status})
                action = action.update({"set_bus": {"lines_or_id": [(id_, 1)], "lines_ex_id": [(id_, 1)]}})
                return action
        return None

    def get_available_topo_actions(self, observation):
        topo_actions = []
        for sub_id in range(self.obs_space.n_sub):
            # ensure the cooldown sub is okay.
            if observation.time_before_cooldown_sub[sub_id] == 0:
                res_list = self.change_bus_actions_by_id(self.action_space, sub_id, observation)
                topo_actions = topo_actions + res_list
        return topo_actions

    def get_available_dispatch_actions(self, observation):
        action_array = []
        max_up = self.obs_space.gen_max_ramp_up
        max_down = self.obs_space.gen_max_ramp_down
        for i in range(len(max_up)):
            if max_up[i] != 0:
                action = self.action_space({"redispatch": [(i, -max_down[i])]})
                action_array.append(action)
                action = self.action_space({"redispatch": [(i, +max_up[i])]})
                action_array.append(action)
        return action_array

    def change_bus_topology_dispatch_emergency_disc(self, observation):  # it has DoNothing action !!!
        tested_actions = [self.action_space({})]
        # substation topology change action
        topo_list = self.get_available_topo_actions(observation)
        tested_actions = tested_actions + topo_list
        # generator redispatch action
        dispatch_list = self.get_available_dispatch_actions(observation)
        tested_actions = tested_actions + dispatch_list

        # condition to break down line
        rho = copy.deepcopy(observation.rho)
        overflow = copy.deepcopy(observation.timestep_overflow)
        # a) deals with soft overflow
        to_disc = (rho >= 1.0) & (overflow == 3)
        # b) disconnect lines on hard overflow
        to_disc[rho > 2.0] = True

        if np.any(to_disc):
            # 主动断线动作
            for id_ in np.where(to_disc)[0]:
                change_status = self.action_space.get_change_line_status_vect()
                change_status[id_] = True
                action = self.action_space({"change_line_status": change_status})
                tested_actions.append(action)
        chosen_action,chosen_rwd = self.choose_best_action_by_simul(observation, tested_actions)
        return chosen_action,chosen_rwd

    def recover_reference_topology(self, observation):  # it has DoNothing action !!!
        ###### Normal handling -- the topo need to play back after attack is over #######
        if len(self.subs_in_attack) and len(self.lines_in_attack) == 0:
            # tested_action = [self.action_space({})]
            tested_action = []
            for sub_id in self.subs_in_attack:
                # ensure the cooldown sub is okay.
                if (observation.time_before_cooldown_sub[sub_id] == 0):
                    sub_topo = observation.state_of(substation_id=sub_id)["topo_vect"]
                    sub_topo = [math.ceil((i + 1) / 3) for i in sub_topo]
                    action = self.action_space({"set_bus": {"substations_id": [(sub_id, sub_topo)]}})
                    tested_action.append(action)
            chosen_action,chosen_rwd = self.choose_best_action_by_simul(observation, tested_action)
            return chosen_action,chosen_rwd
        return None,None

    def try_out_disconnections(self, observation):
        #################################################
        # Have2Discon line to save overflow timestamp.
        # For soft overflow disconnection and hard overflow disconnection, DoNothing will
        # lead to 12 timestep overflow pentality. the powerline will switch to disconnect
        # automatically. so the reward should be same if we inititivelly disconnect it.
        #################################################
        rho = copy.deepcopy(observation.rho)
        overflow = copy.deepcopy(observation.timestep_overflow)
        # set the critiria after checking the backend logic
        # a) deals with soft overflow
        to_disc = (rho >= 1.0) & (overflow == 3)
        # b) disconnect lines on hard overflow
        to_disc[rho > 2.0] = True

        will_disc = (rho >= 1.0) & (overflow == 2)
        # Emergency Overflow
        if np.any(to_disc):
            ##### try to modify the topo or the redisp first if it can reduce the overflow ######
            ###### topo change can also solve the problem potentially. ######
            tested_actions = [self.action_space({})]
            # topo action
            topo_list = self.get_available_topo_actions(observation)
            tested_actions = tested_actions + topo_list
            # dispatch action
            dispatch_list = self.get_available_dispatch_actions(observation)
            tested_actions = tested_actions + dispatch_list
            # disconnect action
            for id_ in np.where(to_disc)[0]:
                change_status = self.action_space.get_change_line_status_vect()
                change_status[id_] = True
                action = self.action_space({"change_line_status": change_status})
                tested_actions.append(action)
            chosen_action,chosen_rwd = self.choose_best_action_by_simul(observation, tested_actions)
            return chosen_action,chosen_rwd
        # Less emergency overflow, still have one timestamp for DoNothing
        elif np.any(will_disc):
            tested_actions = [self.action_space({})]
            # substation topology change action
            topo_list = self.get_available_topo_actions(observation)
            tested_actions = tested_actions + topo_list
            chosen_action,chosen_rwd = self.choose_best_action_by_simul(observation, tested_actions)
            return chosen_action,chosen_rwd
        return None,None
