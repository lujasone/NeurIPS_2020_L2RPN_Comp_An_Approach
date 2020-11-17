from grid2op.Agent import BaseAgent
from .DoubleDuelingDQN import DoubleDuelingDQN as BackupAgent
import numpy as np
from grid2op.dtypes import dt_int, dt_float, dt_bool
import itertools
import math
import copy

SUBs_AREA1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    , 21, 22, 23, 24, 25, 26, 27, 28, 29,30, 31, 112, 113, 114, 116,70,71,72,69,73]  # 41
SUBs_AREA2 = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
    , 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68]  # 36
SUBs_AREA3 = [74,67, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
             89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
             104, 105, 106, 107, 108, 109, 110, 111, 115,117]  # 41

REDISPATCH_AREA1 = [6, 10]  # , 37 on sub68, 38 on sub69, close to boundary
REDISPATCH_AREA2 = [19, 42, 38]
REDISPATCH_AREA3 = [45, 53, 42] # 42 on sub76, close to area2

# line 108: total 20
LINEs_108_AREA1 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 29, 30, 32, 33, 34, 36, 37, 38, 39, 112]
# line 109: total 29
LINEs_109_AREA2 =[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 64, 112]
# line 117: total 23
LINEs_117_AREA3 =[4, 7, 8, 14, 15, 16, 17, 24, 25, 29, 30, 32, 33, 34, 36, 37, 38, 39, 63, 64, 65, 67, 112]
# line 171: total 29
LINEs_171_AREA4 = [22, 23, 41, 44, 45, 46, 47, 48, 49, 50, 53, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 115, 117]
# line 9: total 26
LINEs_9_AREA5 = [23, 41, 44, 45, 46, 47, 48, 49, 50, 53, 64, 65, 67, 68, 69, 70, 73, 74, 75, 76, 77, 79, 80, 81, 115, 117]
# line 12: total 32
LINEs_12_AREA6 =  [23, 41, 44, 45, 46, 47, 48, 49, 50, 53, 64, 65, 67, 68, 69, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 95, 96, 97, 98, 115, 117]
# line 183: total 20
LINEs_183_AREA7 = [29, 36, 37, 46, 48, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 74, 76, 79, 80, 115]
# line 184: total 28
LINEs_184_AREA8 = [23, 37, 41, 44, 45, 46, 47, 48, 49, 50, 53, 63, 64, 65, 67, 68, 69, 70, 73, 74, 75, 76, 77, 79, 80, 81, 115, 117]

LINE_9_SUBs = [48]  #9, 84
LINE_8_SUBs = [99]  #8. 56
LINE_7_SUBs = [11,79]  #7. 35       ,175 totally     3=4
LINE_6_SUBs = [16,36,58,76,68,91] #6    20*6=120  3=3
LINE_5_SUBs = [4,14,31,53,55,69,74,93,104,95]  #5   10*10=100, 2=3
LINE_4_SUBs = [10,29,26,22,18,33,64,61,60,65,84,102,109]


HUB_SUBs = [16,68]

COMPLEX_SUBs = [10,29,26,22,18,33,64,61,60,84,102,109,4,14,31,65,69,74,93,104,95,16,36,53,55,68,11,58,76, 91,79,99,48]

MAX_BUS_CHANGE_SET = 2

REDISPATCH = False
DEBUG = False


class MyAgent(BaseAgent):
    """
    The template to be used to create an agent: any controller of the power grid is expected to be a subclass of this
    grid2op.Agent.BaseAgent.
    """
    def __init__(self, observation_space, action_space):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)
        self.obs_space = observation_space
        self.backupAgent = BackupAgent(observation_space, action_space, name="DDQN_XK")

    def load(self, path):
        self.backupAgent.Qmain.load_network(path)

    def get_gen_list_by_sub_list(self, sub_list, observation):
        gen_list =[]
        for sub_id in sub_list:
            if observation.time_before_cooldown_sub[sub_id]==0:
                if sub_id == 11:
                    gen_list.append(6)
                elif sub_id == 17:    
                    gen_list.append(10)
                elif sub_id == 41:    
                    gen_list.append(19)
                elif sub_id == 69:    
                    gen_list.append(38)                
                elif sub_id == 76:    
                    gen_list.append(42)
                elif sub_id == 82:    
                    gen_list.append(45)                
                elif sub_id == 91:    
                    gen_list.append(51)
                elif sub_id == 99:    
                    gen_list.append(53)                
        return gen_list
            
    def get_subs_of_line(self, line_id):
        sub_list = []
        or_subid = self.action_space.line_or_to_subid[line_id]
        sub_list.append(or_subid)
        ex_subid = self.action_space.line_ex_to_subid[line_id]
        sub_list.append(ex_subid)
        return sub_list

    def retrieve_sub_list_by_line_id(self, line_id, depth):
        # depth = 0
        sub_layer_list = []
        sub_layer_list.append(self.get_subs_of_line(line_id))

        for i in range(depth):
            single_layer_subs = []
            for sub_id in sub_layer_list[-1]:
                lines_list = []
                sub_objs = self.action_space.get_obj_connect_to(substation_id=sub_id)
                or_ids = sub_objs["lines_or_id"]
                ex_ids = sub_objs["lines_ex_id"]
                lines_list.extend(or_ids)
                lines_list.extend(ex_ids)
                for tmp_line_id in lines_list:
                    tmp_sub_list = self.get_subs_of_line(tmp_line_id)
                    single_layer_subs.extend(tmp_sub_list)
            single_layer_subs = list(set(single_layer_subs))
            sub_layer_list.append(single_layer_subs)

        sub_res = list(set(sub_layer_list[-1]))
        return sub_res

    def recover_reference_topology(self, observation):   # it has DoNothing action !!!
        ###### Normal handling -- the topo need to play back after attack is over #######
        if len(self.subs_in_rollback):
            #tested_action = [self.action_space({})]
            tested_action = []
            for sub_id in self.subs_in_rollback:
                # ensure the cooldown sub is okay.
                if (observation.time_before_cooldown_sub[sub_id]==0):
                    sub_topo = observation.state_of(substation_id=sub_id)["topo_vect"]
                    #sub_topo[:] = 1
                    sub_topo = [math.ceil((i + 1) / 3) for i in sub_topo]
                    action = self.action_space({"set_bus": {"substations_id": [(sub_id, sub_topo)]}})
                    tested_action.append(action)
            chosen_action,_ = self.choose_best_action_by_simul(observation, tested_action)
            return chosen_action
        return None

    def get_impacted_subs_areas_by_powerline_list(self, observation):
        # due to big grid, we only handle the critical overflow lines by providing corresponding topos.
        # index sort by rho from large to small.
        sort_rho=-np.sort(-observation.rho)#sort in descending order for positive values
        sort_indices=np.argsort(-observation.rho)
        ltc_list=[sort_indices[i] for i in range(len(sort_rho)) if sort_rho[i]>=1 ]

        # get critical line list to be disconnected.
        # a) deals with soft overflow
        to_disc = (observation.rho >=1.0) & (observation.timestep_overflow == 3)
        # b) disconnect lines on hard overflow
        to_disc[observation.rho >2.0] = True
        ltc_critical = np.where(to_disc)[0]
        # sort by rho among ltc_critical
        if len(ltc_critical)>0:
            ltc_critical = [x for x in ltc_list if x in ltc_critical]

        # get less critical lines
        will_disc = (observation.rho >=1.0) & (observation.timestep_overflow == 2)
        ltc_less_critical = np.where(will_disc)[0]
        # sort by rho among less ltc_critical
        if len(ltc_less_critical)>0:
            ltc_less_critical = [x for x in ltc_list if x in ltc_less_critical]

        # only get the 1st critical line.
        if len(ltc_critical)>0:
            critical_line_id = ltc_critical[0]
        elif len(ltc_less_critical)>0:
            critical_line_id = ltc_less_critical[0]
        else:
            critical_line_id = ltc_list[0]
        
        sub_list = self.retrieve_sub_list_by_line_id(critical_line_id, 2)
        
        return sub_list

    def change_bus_actions_by_id(self, action_space, sub_num, observation):
        action_array = []
        topo_objs = []
        sub_objs = action_space.get_obj_connect_to(substation_id=sub_num)
        or_ids = sub_objs["lines_or_id"]
        ex_ids = sub_objs["lines_ex_id"]
        if len(or_ids)+len(ex_ids) < 3:
            return action_array
       # added by Jason
        if sub_num==37 or sub_num ==70:
            return action_array
        # added end

        for i in range(len(or_ids)):
            if or_ids[i] == 131 or or_ids[i] == 141 or or_ids[i] == 164 or or_ids[i] == 152 or or_ids[i] == 33 or or_ids[i] == 17:
                continue
            else:
                topo_obj = "or_" + str(or_ids[i])
                topo_objs.append(topo_obj)

        for i in range(len(ex_ids)):
            if ex_ids[i] == 131 or ex_ids[i] == 141 or ex_ids[i] == 164 or ex_ids[i] == 152 or ex_ids[i] == 33 or ex_ids[i] == 17:
                continue
            else:
                topo_obj = "ex_" + str(ex_ids[i])
                topo_objs.append(topo_obj)
        for i in range(1, len(topo_objs) + 1):
            iter = itertools.combinations(topo_objs, i)
            iter = list(iter)
            for j in range(len(iter)):
                action_list = list(iter[j])
                load_line = []
                or_line = []
                ex_line = []

                if len(action_list) <= MAX_BUS_CHANGE_SET and len(action_list) > 1:
                    for k in range(len(action_list)):
                        if "or_" in action_list[k]:
                            or_line.append(int(action_list[k][3:]))
                        if "ex_" in action_list[k]:
                            ex_line.append(int(action_list[k][3:]))
                    action = action_space({"change_bus": {"loads_id": load_line, "lines_or_id": or_line, "lines_ex_id": ex_line}})
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
                    if DEBUG:
                        print("illegal!!!")
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

    def compare_with_model(self,act,rwd):
        if rwd > -0.9:
            return act
        chosen_action, chosen_rwd = self.backupAgent.get_action_by_model()
        if chosen_action is not None and chosen_rwd > rwd:
            return chosen_action
        return act

    def act(self, observation, reward, done):
        state = self.backupAgent.convert_obs(observation)
        self.backupAgent._save_current_frame(state)
        self.backupAgent.obs = observation
        self.obs = observation
        ##### LineReconnection ########
        # we try to reconnect a line if possible, THIS IS TOP ACTION.
        action_reco = self.reco_line(observation)
        if action_reco is not None:
            return action_reco

        ###### Other cases.#######################
        line_stat_s = copy.deepcopy(observation.line_status)
        cooldown = copy.deepcopy(observation.time_before_cooldown_line)
        rho = copy.deepcopy(observation.rho)

        disconnected = np.any(~line_stat_s & (cooldown > 0))
        all_connected = np.all(line_stat_s)
        overflowed = np.any(rho >= 1.0)
        no_overflowed = np.all(rho < 1.0)

        time_next_maintenance = observation.time_next_maintenance
        will_maintain = (time_next_maintenance <= 4.0)&(time_next_maintenance > 0)

        ##### TODO: add overflow & will_maintain 
        if np.any(will_maintain) and no_overflowed:
                #get the 1st maintenance line (to be)
                will_maintain_line_list = np.where(will_maintain)[0]
                tmp_amp = 0
                critical_line_id = -1
                for _id in will_maintain_line_list:
                    if (abs(observation.a_or[_id]) > tmp_amp):
                        tmp_amp =abs(observation.a_or[_id])
                        critical_line_id = _id
                
                sub_list = self.retrieve_sub_list_by_line_id(critical_line_id,2)

                topo_actions = [self.action_space({})]
                for sub_id in sub_list:
                    # ensure the cooldown sub is okay.
                    if observation.time_before_cooldown_sub[sub_id]==0:
                        res_list = self.change_bus_actions_by_id(self.action_space, sub_id, observation)
                        topo_actions = topo_actions + res_list
                        
                if REDISPATCH:
                    dispatch_actions = self.get_available_dispatch_actions(sub_list, observation)
                    topo_actions = topo_actions + dispatch_actions

                chosen_action, highest_reward = self.choose_best_action_by_simul(observation, topo_actions)
                chosen_action = self.compare_with_model(chosen_action, highest_reward)
                return chosen_action

        # case3: dicconnected and overflowed
        if disconnected and overflowed:
            action_topo, rwd_topo = self.change_bus_topology_dispatch_emergency_disc(observation)  # from all subs.
            action_topo = self.compare_with_model(action_topo, rwd_topo)
            if action_topo is not None:
                return action_topo
            return self.action_space({})   # DoNothing

        # case 4: disconnected and no overflowed
        elif disconnected and no_overflowed:
            return self.action_space({})   # DoNothing

        # case 5: all-connected and overflowed
        elif all_connected and overflowed:
            action_list = []
            # disconnect lines ourselves.
            action_discon,chosen_rwd = self.try_out_disconnections(observation)
            if action_discon is not None:
                return action_discon
            return self.action_space({})  # DoNothing

        # case 6: all-connected and no_overflowed
        elif all_connected and no_overflowed:
            return self.action_space({})   # DoNothing

        else:
            return self.action_space({})   # DoNothing

    #we reconnect lines when possible
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
                res.append(action)
            # chosen_action = res[0]
            if len(res) == 1:
                return res[0]
            else:    
                chosen_action, chosen_rwd = self.choose_best_action_by_simul(observation, res)
                return chosen_action
        return None

    def get_available_topo_actions(self, observation):
        topo_actions = []
        sub_list = self.get_impacted_subs_areas_by_powerline_list(observation)
        for sub_id in sub_list:
            # ensure the cooldown sub is okay.
            if observation.time_before_cooldown_sub[sub_id]==0:
                res_list = self.change_bus_actions_by_id(self.action_space, sub_id, observation)
                topo_actions = topo_actions + res_list
        return topo_actions, sub_list

    def get_available_dispatch_actions(self, sub_list, observation):
        dispatch_actions = []
        gen_list = self.get_gen_list_by_sub_list(sub_list,observation)

        for gen_id in gen_list:
            action = self.action_space({"redispatch": [(gen_id,
                                        -self.obs_space.gen_max_ramp_down[gen_id])]})
            dispatch_actions.append(action)
            action = self.action_space({"redispatch": [(gen_id,
                                        self.obs_space.gen_max_ramp_up[gen_id])]})
            dispatch_actions.append(action)
        return dispatch_actions

    def change_bus_topology_dispatch_emergency_disc(self, observation):  # it has DoNothing action !!!
        tested_actions = [self.action_space({})]
        # substation topology change action
        topo_list, sub_list = self.get_available_topo_actions(observation)
        tested_actions = tested_actions + topo_list
        if REDISPATCH:
            # generator redispatch action
            dispatch_actions = self.get_available_dispatch_actions(sub_list, observation)
            tested_actions = tested_actions + dispatch_actions

        # change line status action
        rho = copy.deepcopy(observation.rho)
        overflow = copy.deepcopy(observation.timestep_overflow)
        # a) deals with soft overflow
        to_disc = (rho >=1.0) & (overflow == 3)
        # b) disconnect lines on hard overflow
        to_disc[rho >2.0] = True

        if np.any(to_disc):
            for id_ in np.where(to_disc)[0]:
                change_status = self.action_space.get_change_line_status_vect()
                change_status[id_] = True
                action = self.action_space({"change_line_status": change_status})
                tested_actions.append(action)

        chosen_action,chosen_rwd = self.choose_best_action_by_simul(observation, tested_actions)
        return chosen_action,chosen_rwd

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
        to_disc = (rho >=1.0) & (overflow == 3)
        # b) disconnect lines on hard overflow
        to_disc[rho >2.0] = True
        will_disc = (rho >=1.0) & (overflow == 2)

        # Emergency Overflow
        if np.any(to_disc):
            ##### try to modify the topo or the redisp first if it can reduce the overflow ######
            ###### topo change can also solve the problem potentially. ######
            tested_actions = [self.action_space({})]
            # substation topology change action
            topo_list, sub_list = self.get_available_topo_actions(observation)
            tested_actions = tested_actions + topo_list
            if REDISPATCH:
                # generator redispatch action
                dispatch_actions = self.get_available_dispatch_actions(sub_list, observation)
                tested_actions = tested_actions + dispatch_actions

            # change line status action
            for id_ in np.where(to_disc)[0]:
                change_status = self.action_space.get_change_line_status_vect()
                change_status[id_] = True
                action = self.action_space({"change_line_status": change_status})
                tested_actions.append(action)

            chosen_action, chosen_rwd = self.choose_best_action_by_simul(observation, tested_actions)
            return chosen_action,chosen_rwd
        # Less emergency overflow, still have one timestamp for DoNothing
        elif np.any(will_disc):
            tested_actions = [self.action_space({})]
            # substation topology change action 
            topo_list, _ = self.get_available_topo_actions(observation)
            tested_actions = tested_actions + topo_list
            chosen_action, chosen_rwd = self.choose_best_action_by_simul(observation, tested_actions)
            return chosen_action,chosen_rwd

        return None,None