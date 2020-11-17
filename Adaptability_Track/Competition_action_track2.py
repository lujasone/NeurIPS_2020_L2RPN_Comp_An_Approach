###################################################################################################
###################################################################################################
## 	filename: Competition_action.py
## 	description: define the action space of agent_conventer
##               1.define three class of actions with different ways
##               2.you can indentify your action_space by combination of anyone of each action class
## 	version: 0.81
## 	author: kxu
##  change record:     2020.07.07----change the function "set_line_actions" to avoid some illegal actions
##                     2020.07.13----Add the double line stauts change and set actions
##                     2020.08.24----Apply to track2 competition
####################################################################################################

import itertools
MAX_BUS_CHANGE_SET = 3

def change_line_actions(action_space):
    # change line status actions,without line(2) which may cause error
    #  (num:58)
    action_array = []
    line_num = len(action_space.get_change_line_status_vect())
    for i in range(line_num):
        if i == 2:
            continue
        else:
            change_status = action_space.get_change_line_status_vect()
            change_status[i] = True
            action = action_space({"change_line_status": change_status})
            action_array.append(action)

    return action_array

def set_line_actions(action_space):
    # set line status actions
    # (num:59)
    action_array = []
    line_num = len(action_space.get_change_line_status_vect())
    for i in range(line_num):
        set_status = action_space.get_set_line_status_vect()
        set_status[i] = 1
        action = action_space({"set_line_status": set_status})
        action_array.append(action)
    return action_array

def change_bus_actions_with_doublelines_and_maxbusset(action_space):
    # (num:964)
    action_array = []
    for sub_num in range(action_space.n_sub):
        topo_objs = []
        sub_objs = action_space.get_obj_connect_to(substation_id=sub_num)
        or_ids = sub_objs["lines_or_id"]
        ex_ids = sub_objs["lines_ex_id"]
        if len(or_ids)+len(ex_ids) < 3:
            continue
        # added by Jason
        if sub_num==37 or sub_num ==70:
            continue
        # added end    
        for i in range(len(or_ids)):
            if or_ids[i] == 131 or or_ids[i] == 141 or or_ids[i] == 164 or or_ids[i] == 152 or or_ids[i] == 33 or \
                    or_ids[i] == 17:
                continue
            else:
                topo_obj = "or_" + str(or_ids[i])
                topo_objs.append(topo_obj)

        for i in range(len(ex_ids)):
            if ex_ids[i] == 131 or ex_ids[i] == 141 or ex_ids[i] == 164 or ex_ids[i] == 152 or ex_ids[i] == 33 or \
                    ex_ids[i] == 17:
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
                if len(action_list) <= MAX_BUS_CHANGE_SET and len(action_list)>1:
                    for k in range(len(action_list)):
                        if "or_" in action_list[k]:
                            or_line.append(int(action_list[k][3:]))
                        if "ex_" in action_list[k]:
                            ex_line.append(int(action_list[k][3:]))
                    action = action_space({"change_bus": {"loads_id": load_line, "lines_or_id": or_line, "lines_ex_id": ex_line}})
                    action_array.append(action)

    return action_array

def dispatch_gen_actions(action_space):
    action_array = []
    max_up = action_space.gen_max_ramp_up
    max_down = action_space.gen_max_ramp_down
    for i in range(len(max_up)):
        if max_up[i] != 0:
            action = action_space({"redispatch": [(i, -max_down[i])]})
            action_array.append(action)
            action = action_space({"redispatch": [(i, -max_down[i] / 2)]})
            action_array.append(action)
            action = action_space({"redispatch": [(i, +max_up[i] / 2)]})
            action_array.append(action)
            action = action_space({"redispatch": [(i, +max_up[i])]})
            action_array.append(action)

    return action_array

def my_actions(action_space):
    donothing = [action_space({})]
    line_actions = change_line_actions(action_space)
    #bus_actions = change_bus_actions_with_doublelines_and_maxbusset(action_space) + set_line_actions(action_space)
    bus_actions = change_bus_actions_with_doublelines_and_maxbusset(action_space)
    gen_actions = dispatch_gen_actions(action_space)
    #competition_actions = donothing + bus_actions + line_actions + gen_actions
    competition_actions = donothing + bus_actions + line_actions
    
    return competition_actions
