#!/bin/bash

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

export DQN_NAME=dqn-track2
#export DQN_DATA=~/data_grid2op/rte_case14_realistic
#export DQN_DATA=DQN_DATA=/home/ubuntu/anaconda3/envs/grid2op/lib/python3.6/site-packages/grid2op/data/l2rpn_wcci_2020
export DQN_DATA=/home/ubuntu/anaconda3/envs/grid2op_new/lib/python3.6/site-packages/grid2op/data/l2rpn_neurips_2020_track2_small

#./inspect_action_space.py --path_data $DQN_DATA

rm -rf ./logs-train/$DQN_NAME
./train.py\
    --name $DQN_NAME \
    --num_train_steps 1048576 \
    --num_frames 2
