# A winning approach of NeurIPS_2020_L2RPN_Comp
This is a repository for L2RPN NeurIPS 2020 Competition Track1-Robustness and Track2-Adaptability during **Aug 2020 - Nov 2020**. Our approach based on this repository is ranked the 3rd in both competition tracks.

https://l2rpn.chalearn.org

Track1 (Robustness):
https://competitions.codalab.org/competitions/25426

Track2 (Adaptability):
https://competitions.codalab.org/competitions/25427

Grid2Op platform can be found in:
https://github.com/rte-france/Grid2Op

The baseline provided by RTE-France can be found in:
https://github.com/rte-france/l2rpn-baselines

# Summary
The goal of Robustness Track is to develop agent to be robust to unexpected events and keep delivering reliable electricity everywhere even in difficult circumstances. An adversarial opponent will attack some lines of the grid everyday randomly. The agent has to overcome the attack and operate the grid as long as possible, minimize the operation cost including powerlines losses, redispatch cost and blackout cost as penalty.

The goal of Adaptability Track is to develop agent to adapt to new energy productions in the grid with an increasing share of renewable energies which might be less controllable. Operation cost should be considered as well.

This repository presents a value-based RL agent for these two competitions. Two agents will work together with different strategies. It is inspired by the works of Ziming Yan, Yan Xu, Nanyang Technological University (https://github.com/ZM-Learn/L2RPN_WCCI_a_Solution). The Dueling Double DQN agent is revised from L2RPN baseline code. We make much analysis work on rewards combination which is key for RL algorithm. 

# Environment
The code is programmed by Pthon3.7. It runs on Grid2Op platform.  Grid2Op github homepage proivdes installation in detail.

# License
This Source Code is subject to the terms of the GNU Lesser General Public License v3.0. If a copy of the LGPL-v3 was not distributed with this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.html.
 


