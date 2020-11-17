# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import json
import math
import copy
import numpy as np
import tensorflow as tf
from grid2op.Agent import AgentWithConverter

import sys
sys.path.append(os.path.dirname(__file__))

from .DoubleDuelingDQNConfig import DoubleDuelingDQNConfig as cfg
from .DoubleDuelingDQN_NN import DoubleDuelingDQN_NN
from .prioritized_replay_buffer import PrioritizedReplayBuffer
from .Action_Converter import Action_Converter
from grid2op.dtypes import dt_int, dt_float, dt_bool
import itertools
import csv

MAX_BUS_CHANGE_SET = 5
HUB_SUBs = [16]
COMPLEX_SUBs = [26, 4]
RUN_LOGIC_4_MORE_ACTIONS = True

DEFAULT_OUTPUT_NAME = "steps_rewards.csv"

class DoubleDuelingDQN(AgentWithConverter):
    def __init__(self,
                 observation_space,
                 action_space,
                 name=__name__,
                 is_training=False):
        # Call parent constructor
        AgentWithConverter.__init__(self, action_space,
                                    action_space_converter=Action_Converter)
        self.obs_space = observation_space
        self.all_actions = []
        actions = np.load(os.path.split(os.path.realpath(__file__))[0]+'/'+'track1_885_actions.npz', allow_pickle=True)['all_actions']
        for action in actions:
            self.all_actions.append(action_space.from_vect(action))

        # Store constructor params
        self.name = name
        self.num_frames = cfg.N_FRAMES
        self.is_training = is_training
        self.batch_size = cfg.BATCH_SIZE
        self.lr = cfg.LR

        # Declare required vars
        self.Qmain = None
        self.obs = None
        self.state = []
        self.frames = []
        self.lines_in_attack = []
        self.subs_in_attack = []

        self.output_file = DEFAULT_OUTPUT_NAME
        # Declare training vars
        self.per_buffer = None
        self.done = False
        self.frames2 = None
        self.epoch_rewards = None
        self.epoch_alive = None
        self.Qtarget = None
        self.epsilon = 0.0

        self.observation_size = 22+37+59*8+177+36
        self.action_size = self.action_space.size()
        if cfg.VERBOSE:
            print ("self.obs_space.size_obs() = {}".format(self.obs_space.size_obs())) ## Attribute error?
            print ("self.observation_size = {}".format(self.observation_size))
            print ("self.action_size = {}".format(self.action_size))

        # Load network graph
        self.Qmain = DoubleDuelingDQN_NN(self.action_size,
                                         self.observation_size,
                                         num_frames=self.num_frames,
                                         learning_rate=self.lr,
                                         learning_rate_decay_steps=cfg.LR_DECAY_STEPS,
                                         learning_rate_decay_rate=cfg.LR_DECAY_RATE)
        # Setup training vars if needed
        if self.is_training:
            self._init_training()

    def _init_training(self):
        self.epsilon = cfg.INITIAL_EPSILON
        self.frames2 = []
        self.epoch_rewards = []
        self.epoch_alive = []
        self.per_buffer = PrioritizedReplayBuffer(cfg.PER_CAPACITY, cfg.PER_ALPHA)
        self.Qtarget = DoubleDuelingDQN_NN(self.action_size,
                                           self.observation_size,
                                           num_frames = self.num_frames)

    def _reset_state(self, current_obs):
        # Initial state
        self.obs = current_obs
        self.state = self.convert_obs(self.obs)
        self.done = False

    def _reset_frame_buffer(self):
        # Reset frame buffers
        self.frames = []
        if self.is_training:
            self.frames2 = []

    def _save_current_frame(self, state):
        self.frames.append(state.copy())
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)

    def _save_next_frame(self, next_state):
        self.frames2.append(next_state.copy())
        if len(self.frames2) > self.num_frames:
            self.frames2.pop(0)

    def _adaptive_epsilon_decay(self, step):
        ada_div = cfg.DECAY_EPSILON / 10.0
        step_off = step + ada_div
        ada_eps = cfg.INITIAL_EPSILON * -math.log10((step_off + 1) / (cfg.DECAY_EPSILON + ada_div))
        ada_eps_up_clip = min(cfg.INITIAL_EPSILON, ada_eps)
        ada_eps_low_clip = max(cfg.FINAL_EPSILON, ada_eps_up_clip)
        return ada_eps_low_clip

    def _save_hyperparameters(self, logpath, env, steps):
        r_instance = env.reward_helper.template_reward
        hp = {
            "lr": cfg.LR,
            "lr_decay_steps": cfg.LR_DECAY_STEPS,
            "lr_decay_rate": cfg.LR_DECAY_RATE,
            "batch_size": cfg.BATCH_SIZE,
            "stack_frames": cfg.N_FRAMES,
            "iter": steps,
            "e_start": cfg.INITIAL_EPSILON,
            "e_end": cfg.FINAL_EPSILON,
            "e_decay": cfg.DECAY_EPSILON,
            "discount": cfg.DISCOUNT_FACTOR,
            "per_alpha": cfg.PER_ALPHA,
            "per_beta": cfg.PER_BETA,
            "per_capacity": cfg.PER_CAPACITY,
            "update_freq": cfg.UPDATE_FREQ,
            "update_hard": cfg.UPDATE_TARGET_HARD_FREQ,
            "update_soft": cfg.UPDATE_TARGET_SOFT_TAU,
            "reward": dict(r_instance)
        }
        hp_filename = "{}-hypers.json".format(self.name)
        hp_path = os.path.join(logpath, hp_filename)
        with open(hp_path, 'w') as fp:
            json.dump(hp, fp=fp, indent=2)

    ## Agent Interface
    def convert_obs(self, observation):
        tmp_list_vect = ['prod_p','load_p','p_or','a_or','p_ex','a_ex','rho','topo_vect','line_status',
                         'timestep_overflow','time_before_cooldown_line','time_before_cooldown_sub'] # 22+37+59*8+177+36

        li_vect=  []
        for el in tmp_list_vect:
            if el in observation.attr_list_vect:
                v = observation._get_array_from_attr_name(el).astype(np.float32)
                v_fix = np.nan_to_num(v)
                v_norm = np.linalg.norm(v_fix)
                if v_norm > 1e6:
                    v_res = (v_fix / v_norm) * 10.0
                else:
                    v_res = v_fix
                if el =='rho' or el =='topo_vect' or el=='line_status':
                    pass
                else:
                    v_res = v_res/100
                li_vect.append(v_res)
        return np.concatenate(li_vect)

    def convert_act(self, action):
        return super().convert_act(action)

    ## Baseline Interface
    def reset(self, observation):
        self._reset_state(observation)
        self._reset_frame_buffer()

    def my_act(self, state, reward, done=False):
        # Register current state to stacking buffer
        self._save_current_frame(state)

        # We need at least num frames to predict
        if len(self.frames) < self.num_frames:
            return 0 # Do nothing

        # Infer with the last num_frames states
        action = self.common_choose_action()
        a = self.all_actions.index(action)
        return a

    def get_action_by_model(self):
        if len(self.frames) < self.num_frames:
            chosen_action = self.action_space({})
            simul_obs, chosen_rwd, simul_has_error, simul_info = self.obs.simulate(chosen_action)
            return chosen_action,chosen_rwd
        chosen_action_id, q_predictions = self.Qmain.predict_move(np.array(self.frames))
        top_actions = np.argsort(q_predictions)[-1: -4: -1].tolist()  # 选前3个
        top_actions.insert(0, 0)  # Put DoNothing at the 1st position.
        res = [self.convert_act(act_id) for act_id in tuple(top_actions)]
        observation = self.obs
        chosen_action,chosen_rwd = self.choose_best_action_by_simul(observation, res)
        return chosen_action,chosen_rwd

    def infer_choose_action(self):
        observation = self.obs
        ##### LineReconnection ########
        line_stat_s = observation.line_status
        cooldown = observation.time_before_cooldown_line
        can_be_reco = ~line_stat_s & (cooldown == 0)
        if np.any(can_be_reco):
            max_score = float('-inf')
            for id_ in np.where(can_be_reco)[0]:
                change_status = self.action_space.get_change_line_status_vect()
                change_status[id_] = True
                res = self.action_space({"change_line_status": change_status})
                obs_simulate, reward_simulate, done_simulate, info = observation.simulate(res)
                if reward_simulate > max_score:
                    max_score = reward_simulate
                    chosen_action = res
            return chosen_action

        ##### rho  ########
        if np.sum(observation.rho < 0.9) == len(observation.rho):
            return self.action_space({})  # DoNothing # DoNothing
        else:
            chosed_action,chosen_rwd = self.get_action_by_model()
            return chosed_action

    def act(self, observation, reward, done):
        self.obs = observation
        transformed_observation = self.convert_obs(observation)
        encoded_act = self.my_act(transformed_observation, reward, done)
        return self.convert_act(encoded_act)

    def load(self, path):
        self.Qmain.load_network(path)
        if self.is_training:
            self.Qmain.update_target_hard(self.Qtarget.model)

    def save(self, path):
        self.Qmain.save_network(path)

    def change_bus_actions_by_id(self,action_space, sub_num, observation):
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
                max_bus_change_set = MAX_BUS_CHANGE_SET
                if sub_num in HUB_SUBs:
                    max_bus_change_set = 8
                #if len(action_list) <= MAX_BUS_CHANGE_SET and len(action_list) > 1:
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
                    if cfg.VERBOSE:
                        print("illegal!!!")
                    continue
                simul_obs, simul_reward, simul_has_error, simul_info = observation.simulate(action)
                resulting_rewards[i] = simul_reward
            reward_idx = int(np.nanargmax(resulting_rewards))  # rewards.index(max(rewards))
            highest_reward = np.max(resulting_rewards)
            best_action = tested_action[reward_idx]
        #only one action to be done
        elif len(tested_action) == 1:
            best_action = tested_action[0]
            simul_obs, highest_reward, simul_has_error, simul_info = observation.simulate(best_action)
        else:
            best_action = self.action_space({})
            simul_obs, highest_reward, simul_has_error, simul_info = observation.simulate(best_action)
        return best_action,highest_reward

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

    def remove_from_attack_line_list(self,action):
        lines_impacted, subs_impacted = action.get_topological_impact()
        change_lines = np.where(lines_impacted == True)[0]
        if len(change_lines) > 0:
            if change_lines[0] in self.lines_in_attack:
                self.lines_in_attack.remove(change_lines[0])


    def common_choose_action(self):
        observation = self.obs
        self.add_to_attack_line_list(observation)
                
        ##### LineReconnection ########
        # we try to reconnect a line if possible, THIS IS TOP ACTION.
        action_reco,reco_rwd = self.reco_line(observation)
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
            # for all substations we get all possible topology action list, select topo actions.
            action_topo,rwd_topo = self.change_bus_topology_dispatch_emergency_disc(observation) #from all subs.
            chosen_action = self.compare_with_model(action_topo, rwd_topo)
            if chosen_action is not None:
                self.add_to_attack_sub_list(chosen_action)
                return chosen_action
            return self.action_space({})   # DoNothing

        # case 2: attacked & No overflow
        elif attacked and no_overflowed:
            return self.action_space({})   # DoNothing

        # case3: dicconnected (from attacked or system-discon) and overflowed
        elif disconnected and overflowed:
            # same as attacked and overflowed
            action_topo,rwd_topo = self.change_bus_topology_dispatch_emergency_disc(observation) #from all subs.
            chosen_action = self.compare_with_model(action_topo,rwd_topo)
            if chosen_action is not None:
                self.add_to_attack_sub_list(chosen_action)
                return chosen_action
            return self.action_space({})   # DoNothing

        # case 4: disconnected and no overflowed
        elif disconnected and no_overflowed:
            # recover topo due to attack.
            action_recover = self.recover_reference_topology(observation)
            if action_recover is not None:
                self.remove_from_attack_sub_list(action_recover)
                return action_recover
            return self.action_space({})   # DoNothing

        # case 5: all-connected and overflowed
        elif all_connected and overflowed:
            action_list = []
            # disconnect lines ourselves.
            action_discon = self.try_out_disconnections(observation)
            if action_discon is not None:
                action_list.append(action_discon)

            # recover topo due to attack.
            action_recover = self.recover_reference_topology(observation)
            if action_recover is not None:
                action_list.append(action_recover)

            # select the best action from 2 candidated actions
            chosen_action,highest_reward = self.choose_best_action_by_simul(observation, action_list)
            if (chosen_action == action_recover):
                self.remove_from_attack_sub_list(action_recover)
            return chosen_action

        # case 6: all-connected and no_overflowed
        elif all_connected and no_overflowed:
            # recover topo due to attack.
            action_recover = self.recover_reference_topology(observation)
            if action_recover is not None:
                self.remove_from_attack_sub_list(action_recover)
                return action_recover
            return self.action_space({})   # DoNothing

        else:
            return self.action_space({})

    def compare_with_model(self,act,rwd):
        if rwd > 0:
            return act
        observation = self.obs
        if self.is_training:
            if len(self.frames) < self.num_frames:
                chosen_action = self.action_space({})
                return chosen_action
            a = self.Qmain.random_move()
            chosen_action = self.convert_act(a)  # random action
            is_legal = self.check_cooldown_legal(observation, chosen_action)
            while is_legal is False:
                a = self.Qmain.random_move()
                chosen_action = self.convert_act(a)              # random action
                is_legal = self.check_cooldown_legal(observation, chosen_action)
            return chosen_action

        else:
            chosen_action_id, q_predictions = self.Qmain.predict_move(np.array(self.frames))
            top_actions = np.argsort(q_predictions)[-1: -50: -1].tolist()
            top_actions.insert(0, 0)  # Put DoNothing at the 1st position.

            res = [self.convert_act(act_id) for act_id in tuple(top_actions)]
            observation = self.obs
            chosen_action,chosen_rwd = self.choose_best_action_by_simul(observation, res)
            if chosen_rwd > rwd:
                return chosen_action
            return act

    #we reconnect lines that were in maintenance or attacked or normal state when possible
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
                if self.is_training is False:
                    action = action.update({"set_bus": {"lines_or_id": [(id_, 1)], "lines_ex_id": [(id_, 1)]}})
                res.append(action)
            chosen_action,rwd = self.choose_best_action_by_simul(observation, res)
            return chosen_action,rwd
        return None,None
        
    def get_available_topo_actions(self, observation):
        topo_actions = []
        for sub_id in range(self.obs_space.n_sub):
            # ensure the cooldown sub is okay.
            if observation.time_before_cooldown_sub[sub_id]==0:
                res_list = self.change_bus_actions_by_id(self.action_space, sub_id, observation)
                topo_actions = topo_actions + res_list
        return topo_actions

    def get_available_dispatch_actions(self, observation):
        action_array = []
        max_up = self.action_space.gen_max_ramp_up
        max_down = self.action_space.gen_max_ramp_down
        for i in range(len(max_up)):
            if max_up[i] != 0:
                action = self.action_space({"redispatch": [(i, -max_down[i])]})
                action_array.append(action)
                action = self.action_space({"redispatch": [(i, -max_down[i] / 2)]})
                action_array.append(action)
                action = self.action_space({"redispatch": [(i, +max_up[i] / 2)]})
                action_array.append(action)
                action = self.action_space({"redispatch": [(i, +max_up[i])]})
                action_array.append(action)
        return action_array
                       
    def change_bus_topology_dispatch_emergency_disc(self, observation):  # it has DoNothing action !!!
        tested_actions = [self.action_space({})]
        # substation topoloy change action
        topo_list = self.get_available_topo_actions(observation) 
        tested_actions = tested_actions + topo_list
        # generator redispatch action
        dispatch_list = self.get_available_dispatch_actions(observation) 
        tested_actions = tested_actions + dispatch_list
        
        # deal with break down line
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
    
    def recover_reference_topology(self, observation):   # it has DoNothing action !!!     
        ###### Normal handling -- the topo need to play back after attack is over #######
        if len(self.subs_in_attack) and len(self.lines_in_attack)==0:
            #tested_action = [self.action_space({})]
            tested_action = []
            for sub_id in self.subs_in_attack:
                # ensure the cooldown sub is okay.
                if (observation.time_before_cooldown_sub[sub_id]==0):
                    sub_topo = observation.state_of(substation_id=sub_id)["topo_vect"]
                    if self.is_training:
                        if all(sub_topo != -1):
                            sub_topo[:] = 1
                    else:
                        sub_topo = [math.ceil((i + 1) / 3) for i in sub_topo]
                    action = self.action_space({"set_bus": {"substations_id": [(sub_id, sub_topo)]}})
                    tested_action.append(action)
                    
            chosen_action,chosen_rwd = self.choose_best_action_by_simul(observation, tested_action)
            return chosen_action
        return None
    
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
            topo_list = self.get_available_topo_actions(observation) 
            tested_actions = tested_actions + topo_list
            # generator redispatch action
            dispatch_list = self.get_available_dispatch_actions(observation) 
            tested_actions = tested_actions + dispatch_list
            # line status change action
            for id_ in np.where(to_disc)[0]:
                change_status = self.action_space.get_change_line_status_vect()
                change_status[id_] = True
                action = self.action_space({"change_line_status": change_status})  
                tested_actions.append(action)
            chosen_action,chosen_rwd = self.choose_best_action_by_simul(observation, tested_actions)
            return chosen_action
        # Less emergency overflow, still have one timestamp for DoNothing
        elif np.any(will_disc):
            tested_actions = [self.action_space({})]
            topo_list = self.get_available_topo_actions(observation) 
            tested_actions = tested_actions + topo_list            
            chosen_action,chosen_rwd = self.choose_best_action_by_simul(observation, tested_actions)
            return chosen_action            
        return None    

    ## Training Procedure
    def train(self, env,
              iterations,
              save_path,
              logdir = "logs-train"):

        # Loop vars
        num_steps = iterations
        step = 0
        self.epsilon = cfg.INITIAL_EPSILON
        alive_steps = 0
        total_reward = 0
        self.done = True

        # Create file system related vars
        logpath = os.path.join(logdir, self.name)
        os.makedirs(save_path, exist_ok=True)
        modelpath = os.path.join(save_path, self.name + ".h5")
        self.tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        self._save_hyperparameters(save_path, env, num_steps)

        # Training loop
        while step < num_steps:
            # Init first time or new episode
            if self.done:
                new_obs = env.reset()
                self.reset(new_obs)
            if step % 1000 == 0:
                print("Step [{}] -- Random [{}]".format(step, self.epsilon))

            # Choose an action (valid action)
            if np.random.rand(1) < self.epsilon:
                act = self.common_choose_action()
            elif len(self.frames) < self.num_frames:
                act = self.action_space({})  # do nothing
            else:
                chosen_action_id, q_predictions = self.Qmain.predict_move(np.array(self.frames))
                top_actions = np.argsort(q_predictions)[-1: -2: -1].tolist()
                top_actions.insert(0,0)
                res = [self.convert_act(act_id) for act_id in tuple(top_actions)]
                observation = self.obs
                act,highest_reward = self.choose_best_action_by_simul(observation, res)
       
            # Execute action
            new_obs, reward, self.done, info = env.step(act)
            if self.done and len(info['exception'])!=0:
                reward = -100
            new_state = self.convert_obs(new_obs)
            if info["is_illegal"]:
                if cfg.VERBOSE:
                    print (" $$$$$ illegal !!! select action {}".format(a))
            # Only 'change_topo' actions sample can be saved
            self._save_current_frame(self.state)
            a = self.all_actions.index(act)
            if cfg.VERBOSE:
                print("------------------ Actual reward {:.3f},  Actual act:   {},".format(reward, a))
            self._save_next_frame(new_state)
            # Save to experience buffer
            if len(self.frames2) == self.num_frames:
                self.per_buffer.add(np.array(self.frames), a, reward,np.array(self.frames2), self.done)

            # Decay chance of random action
            self.epsilon = self._adaptive_epsilon_decay(step)

            # Perform training at given frequency
            if step % cfg.UPDATE_FREQ == 0 and  \
               len(self.per_buffer) >= self.batch_size:
                # Perform training
                self._batch_train(step)

                if cfg.UPDATE_TARGET_SOFT_TAU > 0.0:
                    tau = cfg.UPDATE_TARGET_SOFT_TAU
                    # Update target network towards primary network
                    self.Qmain.update_target_soft(self.Qtarget.model, tau)

            # Every UPDATE_TARGET_HARD_FREQ trainings, update target completely
            if cfg.UPDATE_TARGET_HARD_FREQ > 0 and \
               step % (cfg.UPDATE_FREQ * cfg.UPDATE_TARGET_HARD_FREQ) == 0:
                self.Qmain.update_target_hard(self.Qtarget.model)

            total_reward += reward
            if self.done:
                self.epoch_rewards.append(total_reward)
                self.epoch_alive.append(alive_steps)
                if cfg.VERBOSE:
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~  Survived [{}] steps".format(alive_steps))
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~~~~  Total reward [{}]".format(total_reward))
                alive_steps = 0
                total_reward = 0
            else:
                alive_steps += 1

            # Save the network every 1000 iterations
            if step > 0 and step % 1000 == 0:
                self.save(modelpath)

            # Iterate to next loop
            step += 1
            # Make new obs the current obs
            self.obs = new_obs
            self.state = new_state

        # Save model after all steps
        self.save(modelpath)
        with open(self.output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.epoch_alive)
            writer.writerow(self.epoch_rewards)

    def _batch_train(self, step):
        # Sample from experience buffer
        sample_batch = self.per_buffer.sample(self.batch_size, cfg.PER_BETA)
        s_batch = sample_batch[0]
        a_batch = sample_batch[1]
        r_batch = sample_batch[2]
        s2_batch = sample_batch[3]
        d_batch = sample_batch[4]
        w_batch = sample_batch[5]
        idx_batch = sample_batch[6]

        Q = np.zeros((self.batch_size, self.action_size))

        # Reshape frames to 1D
        input_size = self.observation_size * self.num_frames
        input_t = np.reshape(s_batch, (self.batch_size, input_size))
        input_t_1 = np.reshape(s2_batch, (self.batch_size, input_size))

        # Save the graph just the first time
        if step == 0:
            tf.summary.trace_on()

        # T Batch predict
        Q = self.Qmain.model.predict(input_t, batch_size = self.batch_size)

        ## Log graph once and disable graph logging
        if step == 0:
            with self.tf_writer.as_default():
                tf.summary.trace_export(self.name + "-graph", step)

        # T+1 batch predict
        Q1 = self.Qmain.model.predict(input_t_1, batch_size=self.batch_size)
        Q2 = self.Qtarget.model.predict(input_t_1, batch_size=self.batch_size)

        # Compute batch Qtarget using Double DQN
        for i in range(self.batch_size):
            doubleQ = Q2[i, np.argmax(Q1[i])]
            Q[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                Q[i, a_batch[i]] += cfg.DISCOUNT_FACTOR * doubleQ

        # Batch train
        loss = self.Qmain.train_on_batch(input_t, Q, w_batch)

        # Update PER buffer
        priorities = self.Qmain.batch_sq_error
        # Can't be zero, no upper limit
        priorities = np.clip(priorities, a_min=1e-8, a_max=None)
        self.per_buffer.update_priorities(idx_batch, priorities)               # 

        # Log some useful metrics every even updates
        if step % (cfg.UPDATE_FREQ * 2) == 0:
            with self.tf_writer.as_default():
                mean_reward = np.mean(self.epoch_rewards)
                mean_alive = np.mean(self.epoch_alive)
                if len(self.epoch_rewards) >= 100:
                    mean_reward_100 = np.mean(self.epoch_rewards[-100:])
                    mean_alive_100 = np.mean(self.epoch_alive[-100:])
                else:
                    mean_reward_100 = mean_reward
                    mean_alive_100 = mean_alive
                tf.summary.scalar("mean_reward", mean_reward, step)
                tf.summary.scalar("mean_alive", mean_alive, step)
                tf.summary.scalar("mean_reward_100", mean_reward_100, step)
                tf.summary.scalar("mean_alive_100", mean_alive_100, step)
                tf.summary.scalar("loss", loss, step)
                tf.summary.scalar("lr", self.Qmain.train_lr, step)
            if cfg.VERBOSE:
                print("loss =", loss)
