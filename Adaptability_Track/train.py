#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import tensorflow as tf

from DoubleDuelingDQN import DoubleDuelingDQN as D3QNAgent
from DoubleDuelingDQNConfig import DoubleDuelingDQNConfig as D3QNConfig
from MyCombinedScaledReward import *

DEFAULT_NAME = "DoubleDuelingDQN"
DEFAULT_SAVE_DIR = "./models"
DEFAULT_LOG_DIR = "./logs-train"
DEFAULT_TRAIN_STEPS = 520000
DEFAULT_N_FRAMES = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-5
DEFAULT_VERBOSE = True

def cli():
    parser = argparse.ArgumentParser(description="Train baseline DDQN")
    # Paths
    parser.add_argument("--name", default=DEFAULT_NAME,
                        help="The name of the model")
    parser.add_argument("--data_dir", default="/home/ubuntu/anaconda3/envs/grid2op_new/lib/python3.6/site-packages/grid2op/data/l2rpn_neurips_2020_track2_small/",
                        help="Path to the dataset root directory")
    parser.add_argument("--save_dir", required=False,
                        default=DEFAULT_SAVE_DIR, type=str,
                        help="Directory where to save the model")
    parser.add_argument("--load_file", required=False,
                        help="Path to model.h5 to resume training with")
    parser.add_argument("--logs_dir", required=False,
                        default=DEFAULT_LOG_DIR, type=str,
                        help="Directory to save the logs")
    parser.add_argument("--num_train_steps", required=False,
                        default=DEFAULT_TRAIN_STEPS, type=int,
                        help="Number of training iterations")
    parser.add_argument("--num_frames", required=False,
                        default=DEFAULT_N_FRAMES, type=int,
                        help="Number of stacked states to use during training")
    parser.add_argument("--batch_size", required=False,
                        default=DEFAULT_BATCH_SIZE, type=int,
                        help="Mini batch size (defaults to 1)")
    parser.add_argument("--learning_rate", required=False,
                        default=DEFAULT_LR, type=float,
                        help="Learning rate for the Adam optimizer")
    return parser.parse_args()


def train(env,
          name = DEFAULT_NAME,
          iterations = DEFAULT_TRAIN_STEPS,
          save_path = DEFAULT_SAVE_DIR,
          load_path = None,
          logs_path = DEFAULT_LOG_DIR,
          num_frames = DEFAULT_N_FRAMES,
          batch_size= DEFAULT_BATCH_SIZE,
          learning_rate= DEFAULT_LR,
          verbose=DEFAULT_VERBOSE):

    # Set config
    D3QNConfig.LR = learning_rate
    D3QNConfig.N_FRAMES = num_frames
    D3QNConfig.BATCH_SIZE = batch_size
    D3QNConfig.VERBOSE = verbose

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    agent = D3QNAgent(env.observation_space,
                      env.action_space,
                      name=name,
                      is_training=True)

    if load_path is not None:
        agent.load(load_path)

    agent.train(env,
                iterations,
                save_path,
                logs_path)


if __name__ == "__main__":
    from grid2op.MakeEnv import make
    from grid2op.Reward import *
    from grid2op.Action import *
    from MyRewards import *
    try:
        from lightsim2grid.LightSimBackend import LightSimBackend
        backend = LightSimBackend()
    except:
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()
        
        
    args = cli()

    env = make(dataset='l2rpn_neurips_2020_track2',
               backend = backend,
               action_class=TopologyAndDispatchAction,
               reward_class=MyCombinedScaledReward
               )

    # Only load 128 steps in ram
    env.chronics_handler.set_chunk_size(128)

    train(env,
          name = args.name,
          iterations = args.num_train_steps,
          save_path = args.save_dir,
          load_path = args.load_file,
          logs_path = args.logs_dir,
          num_frames = args.num_frames,
          batch_size = args.batch_size,
          learning_rate = args.learning_rate)
