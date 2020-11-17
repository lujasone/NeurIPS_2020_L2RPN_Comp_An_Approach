# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import os
import numpy as np

from grid2op.Action import BaseAction
from grid2op.Converter.Converters import Converter
from grid2op.dtypes import dt_float
from Competition_action_track2 import *

class Action_Converter(Converter):
    def __init__(self, action_space):
        Converter.__init__(self, action_space)
        self.__class__ = Action_Converter.init_grid(action_space)
        self.all_actions = my_actions(action_space)
        self.n = 1
        self._init_size = action_space.size()
        self.kwargs_init = {}

    def init_converter(self, **kwargs):
        self.kwargs_init = kwargs
        if self.all_actions is None:
            self.all_actions = []
            # add the do nothing action, always
            self.all_actions.append(super().__call__())
            if "_set_line_status" in self._template_act.attr_list_vect:
                # lines 'set'
                include_ = True
                if "set_line_status" in kwargs:
                    include_ = kwargs["set_line_status"]
                if include_:
                    self.all_actions += self.get_all_unitary_line_set(self)

            if "_switch_line_status" in self._template_act.attr_list_vect:
                # lines 'change'
                include_ = True
                if "change_line_status" in kwargs:
                    include_ = kwargs["change_line_status"]
                if include_:
                    self.all_actions += self.get_all_unitary_line_change(self)

            if "_set_topo_vect" in self._template_act.attr_list_vect:
                # topologies 'set'
                include_ = True
                if "set_topo_vect" in kwargs:
                    include_ = kwargs["set_topo_vect"]
                if include_:
                    self.all_actions += self.get_all_unitary_topologies_set(self)

            if "_change_bus_vect" in self._template_act.attr_list_vect:
                # topologies 'change'
                include_ = True
                if "change_bus_vect" in kwargs:
                    include_ = kwargs["change_bus_vect"]
                if include_:
                    self.all_actions += self.get_all_unitary_topologies_change(self)

            if "_redispatch" in self._template_act.attr_list_vect:
                # redispatch (transformed to discrete variables)
                include_ = True
                if "redispatch" in kwargs:
                    include_ = kwargs["redispatch"]
                if include_:
                    self.all_actions += self.get_all_unitary_redispatch(self)
        elif isinstance(self.all_actions, str):
            # load the path from the path provided
            if not os.path.exists(self.all_actions):
                raise FileNotFoundError("No file located at \"{}\" where the actions should have been stored."
                                        "".format(self.all_actions))
            try:
                all_act = np.load(self.all_actions)
            except Exception as e:
                raise RuntimeError("Impossible to load the data located at \"{}\" with error\n{}."
                                   "".format(self.all_actions, e))
            try:
                self.all_actions = np.array([self.__call__() for _ in all_act])
                for i, el in enumerate(all_act):
                    self.all_actions[i].from_vect(el)
            except Exception as e:
                raise RuntimeError("Impossible to convert the data located at \"{}\" into valid grid2op action. "
                                   "The error was:\n{}".format(self.all_actions, e))
        elif isinstance(self.all_actions, (list, np.ndarray)):
            # assign the action to my actions
            possible_act = self.all_actions[0]
            if isinstance(possible_act, BaseAction):
                self.all_actions = np.array(self.all_actions)
            else:
                try:
                    self.all_actions = np.array([self.__call__() for _ in self.all_actions])
                    for i, el in enumerate(self.all_actions):
                        self.all_actions[i].from_vect(el)
                except Exception as e:
                    raise RuntimeError("Impossible to convert the data provided in \"all_actions\" into valid "
                                       "grid2op action. The error was:\n{}".format(e))
        else:
            raise RuntimeError("Impossible to load the action provided.")
        self.n = len(self.all_actions)

    def filter_action(self, filtering_fun):
        self.all_actions = np.array([el for el in self.all_actions if filtering_fun(el)])
        self.n = len(self.all_actions)

    def save(self, path, name="action_space_vect.npy"):
        if not os.path.exists(path):
            raise FileNotFoundError("Impossible to save the action space as the directory \"{}\" does not exist."
                                    "".format(path))
        if not os.path.isdir(path):
            raise NotADirectoryError("The path to save the action space provided \"{}\" is not a directory."
                                     "".format(path))
        saved_npy = np.array([el.to_vect() for el in self.all_actions]).astype(dtype=dt_float).reshape(self.n, -1)
        np.save(file=os.path.join(path, name), arr=saved_npy)

    def sample(self):
        idx = self.space_prng.randint(0, self.n)
        return idx

    def convert_act(self, encoded_act):
        return self.all_actions[encoded_act]
