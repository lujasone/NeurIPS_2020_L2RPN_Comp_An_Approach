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
from grid2op.Converter import IdToAct

import sys
sys.path.append(os.path.dirname(__file__))

from .DoubleDuelingDQNConfig import DoubleDuelingDQNConfig as cfg
from .DoubleDuelingDQN_NN import DoubleDuelingDQN_NN
from .prioritized_replay_buffer import PrioritizedReplayBuffer
from .Action_Converter import Action_Converter

from grid2op.dtypes import dt_int, dt_float, dt_bool
import itertools
import csv

# added by Jason
from .Competition_action_track2 import *

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

PGM_MAX_BUS_CHANGE_SET = 2

REDISPATCH = True
DEBUG = False
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
        """
        #须与Action_Converter中的all_actions保持一致
        self.all_actions = []
        actions = np.load(os.path.split(os.path.realpath(__file__))[0] + '/' + 'track2_actions.npz', allow_pickle=True)[
            'all_actions']
        for action in actions:
            self.all_actions.append(action_space.from_vect(action))
        """
        self.all_actions = my_actions(action_space)
        print(len(self.all_actions))
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

        self.observation_size = 2300   ##track2
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

    def _filter_action(self, action):
        MAX_ELEM = 2
        act_dict = action.impact_on_objects()
        elem = 0
        elem += act_dict["force_line"]["reconnections"]["count"]
        elem += act_dict["force_line"]["disconnections"]["count"]
        elem += act_dict["switch_line"]["count"]
        elem += len(act_dict["topology"]["bus_switch"])
        elem += len(act_dict["topology"]["assigned_bus"])
        elem += len(act_dict["topology"]["disconnect_bus"])
        elem += len(act_dict["redispatch"]["generators"])

        if elem <= MAX_ELEM:
            return True
        return False

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

    def normalize(self, state):
        #normalize the state
        for i, x in enumerate(state):
            if x >= 1e5: state[i] /= 1e4
            elif x >= 1e4 and x < 1e5 : state[i] /= 1e3
            elif x >= 1e3 and x < 1e4 : state[i] /= 1e2

        return(state)

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
        action = self.common_choose_action(model="test")
        a = self.all_actions.index(action)
        return a

    def get_action_by_model(self):
        if len(self.frames) < self.num_frames:
            chosen_action = self.action_space({})
            simul_obs, chosen_rwd, simul_has_error, simul_info = self.obs.simulate(chosen_action)
            return chosen_action, chosen_rwd
        chosen_action_id, q_predictions = self.Qmain.predict_move(np.array(self.frames))
        top_actions = np.argsort(q_predictions)[-1: -5: -1].tolist()
        # top_actions.insert(0, 0)  # Put DoNothing at the 1st position.

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
            # return get_action_by_model()

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

    def common_choose_action(self,model):
        observation = self.obs
        action_reco, _ = self.reco_line(observation)
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
        will_maintain = (time_next_maintenance <= 6.0) & (time_next_maintenance > 0)

        ##### TODO: add overflow & will_maintain ???
        if np.any(will_maintain) and no_overflowed:
            # get the 1st maintenance line (to be)
            will_maintain_line_list = np.where(will_maintain)[0]
            tmp_amp = 0
            critical_line_id = -1
            for _id in will_maintain_line_list:
                if (abs(observation.a_or[_id]) > tmp_amp):
                    tmp_amp = abs(observation.a_or[_id])
                    critical_line_id = _id
            sub_list = self.retrieve_sub_list_by_line_id(critical_line_id, 2)

            topo_actions = [self.action_space({})]
            # topo_actions = []
            for sub_id in sub_list:
                # ensure the cooldown sub is okay.
                if observation.time_before_cooldown_sub[sub_id] == 0:
                    res_list = self.change_bus_actions_by_id(self.action_space, sub_id, observation)
                    topo_actions = topo_actions + res_list
            if REDISPATCH:
                dispatch_actions = self.get_available_dispatch_actions(sub_list, observation)
                topo_actions = topo_actions + dispatch_actions

            chosen_action, highest_reward= self.choose_best_action_by_simul(observation, topo_actions)
            chosen_action = self.compare_with_model(chosen_action, highest_reward)
            return chosen_action

        # case3: dicconnected and overflowed
        if disconnected and overflowed:

            action_topo, rwd_topo = self.change_bus_topology_dispatch_emergency_disc(observation)  # from all subs.
            action_topo = self.compare_with_model(action_topo, rwd_topo)
            if action_topo is not None:
                return action_topo
            return self.action_space({})  # DoNothing

        # case 4: disconnected and no overflowed
        elif disconnected and no_overflowed:
            return self.action_space({})  # DoNothing

        # case 5: all-connected and overflowed
        elif all_connected and overflowed:
            ## disconnect lines ourselves.
            action_list = []
            action_discon = self.try_out_disconnections(observation)
            if action_discon is not None:
                action_list.append(action_discon)

            # select the best action from 2 candidated actions
            chosen_action,highest_reward= self.choose_best_action_by_simul(observation, action_list)
            if chosen_action is not None:
                return chosen_action
            return self.action_space({})  # DoNothing

        # case 6: all-connected and no_overflowed
        elif all_connected and no_overflowed:
            return self.action_space({})  # DoNothing
        else:
            return self.action_space({})                         # DoNothing

    def compare_with_model(self, act, rwd):
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
                chosen_action = self.convert_act(a)  # random action
                is_legal = self.check_cooldown_legal(observation, chosen_action)
            return chosen_action

        else:
            chosen_action_id, q_predictions = self.Qmain.predict_move(np.array(self.frames))
            top_actions = np.argsort(q_predictions)[
                          -1: -30: -1].tolist()  #  top_actions.insert(0, 0)  # Put DoNothing at the 1st position.

            res = [self.convert_act(act_id) for act_id in tuple(top_actions)]
            observation = self.obs
            chosen_action, chosen_rwd = self.choose_best_action_by_simul(observation, res)
            if chosen_rwd > rwd:
                return chosen_action
            return act
            
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
                if cfg.VERBOSE:
                    print("Step [{}] -- Random [{}]".format(step, self.epsilon))

            # Choose an action (valid action)
            if np.random.rand(1) < self.epsilon:
                act = self.common_choose_action(model = "train")
            elif len(self.frames) < self.num_frames:
                act = self.action_space({})  # do nothing
            else:
                chosen_action_id, q_predictions = self.Qmain.predict_move(np.array(self.frames))
                top_actions = np.argsort(q_predictions)[-1: -3: -1].tolist()
                top_actions.insert(0,0)
                res = [self.convert_act(act_id) for act_id in tuple(top_actions)]
                observation = self.obs
                act,highest_reward = self.choose_best_action_by_simul(observation, res)

            new_obs, reward, self.done, info = env.step(act)
            if self.done and len(info['exception'])!=0:
                reward = -100

            new_state = self.convert_obs(new_obs)
            if info["is_illegal"]:
                if cfg.VERBOSE:
                    print (" $$$$$ illegal !!! select action {}".format(a))
            # Only 'change_topo' actions sample can be saved

            self._save_current_frame(self.state)
            # print(act)
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
            if step % cfg.UPDATE_FREQ == 0 and len(self.per_buffer) >= self.batch_size:
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
        """Trains network to fit given parameters"""
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

    def get_subs_of_line(self, line_id):
        sub_list = []
        or_subid = self.action_space.line_or_to_subid[line_id]
        sub_list.append(or_subid)
        ex_subid = self.action_space.line_ex_to_subid[line_id]
        sub_list.append(ex_subid)
        return sub_list

    #def retrieve_sub_list_by_line_id(self, line_id, observation, depth):
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
            tested_action = []
            for sub_id in self.subs_in_rollback:
                # ensure the cooldown sub is okay.
                if (observation.time_before_cooldown_sub[sub_id]==0):
                    sub_topo = observation.state_of(substation_id=sub_id)["topo_vect"]
                    sub_topo = [math.ceil((i + 1) / 3) for i in sub_topo]
                    action = self.action_space({"set_bus": {"substations_id": [(sub_id, sub_topo)]}})
                    tested_action.append(action)
            chosen_action, chosen_rwd = self.choose_best_action_by_simul(observation, tested_action)
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

                if len(action_list) <= PGM_MAX_BUS_CHANGE_SET and len(action_list) > 1:
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
        else:
            best_action = self.action_space({})
            simul_obs, highest_reward, simul_has_error, simul_info = observation.simulate(best_action)
        # else:
        #     best_action = None
        return best_action,highest_reward
        # we reconnect lines when possible

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
            chosen_action, rwd = self.choose_best_action_by_simul(observation, res)
            return chosen_action, rwd
        return None, None

    def get_available_topo_actions(self, observation):
        topo_actions = []
        sub_list = self.get_impacted_subs_areas_by_powerline_list(observation)
        for sub_id in sub_list:
            # ensure the cooldown sub is okay.
            if observation.time_before_cooldown_sub[sub_id] == 0:
                res_list = self.change_bus_actions_by_id(self.action_space, sub_id, observation)
                topo_actions = topo_actions + res_list
        return topo_actions, sub_list

    def get_available_dispatch_actions(self, sub_list, observation):
        dispatch_actions = []
        gen_list = self.get_gen_list_by_sub_list(sub_list, observation)

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
        # generator redispatch action
        if REDISPATCH:
            dispatch_actions = self.get_available_dispatch_actions(sub_list, observation)
            tested_actions = tested_actions + dispatch_actions
        rho = copy.deepcopy(observation.rho)
        overflow = copy.deepcopy(observation.timestep_overflow)
        # a) deals with soft overflow
        to_disc = (rho >= 1.0) & (overflow == 3)
        # b) disconnect lines on hard overflow
        to_disc[rho > 2.0] = True

        if np.any(to_disc):
            # change line status action
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
        to_disc = (rho >= 1.0) & (overflow == 3)
        # b) disconnect lines on hard overflow
        to_disc[rho > 2.0] = True

        will_disc = (rho >= 1.0) & (overflow == 2)

        # Emergency Overflow
        if np.any(to_disc):
            ##### try to modify the topo or the redisp first if it can reduce the overflow ######
            ###### topo change can also solve the problem potentially. ######
            tested_actions = [self.action_space({})]
            #  substation topology change action
            topo_list, _ = self.get_available_topo_actions(observation)
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
            return chosen_action
        # Less emergency overflow, still have one timestamp for DoNothing
        elif np.any(will_disc):
            tested_actions = [self.action_space({})]
            # substation topology change action
            topo_list, _ = self.get_available_topo_actions(observation)
            tested_actions = tested_actions + topo_list
            chosen_action, chosen_rwd = self.choose_best_action_by_simul(observation, tested_actions)
            return chosen_action

        return None
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