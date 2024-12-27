import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
from permanents import glynn

# device = torch.device("mps"  torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

n_sessions = 100 # number of sessions per iteration
Learning_rate = 0.001 # learning rate, increase this to converge faster
percentile = 95 # top 100-x percentile the agent will learn from
super_percentile = 95 # top 100-x percentile of that survive to next generation

copy_time = 0
add_time = 0
count_time = 0

# helper function to add 3 of the 5 free points present in all 312-avoiding matrices
def add_free_points(input_state, n, step):
    action_taken = torch.zeros([len(input_state)])
    cur_state = torch.clone(input_state)

    if step == 1:
        action_index = 0
    elif step == 2:
        action_index = n**2-n
    else:
        action_index = n**2-1

    cur_state[action_index] = 1
    action_taken[action_index] = 1

    return cur_state, action_taken    
    
# helper function to add corners based on probability on the output of the corner_agent neural network
def add_corner(input_state, action_vec, n, row_boundary, col_boundary):
    corner_added = False
    action_taken = torch.zeros([len(action_vec)])
    cur_state = torch.clone(input_state)

    terminal = False

    while not corner_added:
        action_index = torch.multinomial(action_vec, 1).item()
        action_row = action_index//n
        action_col = action_index%n

        if (action_row == n-2 and action_col != n-1) or (action_col == 1 and action_col != 0):
            action_vec[action_index] = 0
            action_vec = action_vec / torch.sum(action_vec)
        elif row_boundary <= action_row < n-1 and col_boundary <= action_col:
            cur_state[action_index] = 1
            action_taken[action_index] = 1
            corner_added = True
        else:
            action_vec[action_index] = 0
            action_vec = action_vec / torch.sum(action_vec)

    if action_col == n-1:
        terminal = True

    return cur_state, action_taken, terminal, action_row, action_col

# defines a helper function to add point, action_vec is the output, output = agent(cur_state), agent = net
def add_point(input_state, action_vec, forbidden_state, corners, n):
    
    ## add time
    point_added = False
    action_taken = torch.zeros([len(action_vec)])
    cur_state = torch.clone(input_state)
    cur_forbidden = torch.clone(forbidden_state)

    while not point_added:
        action_index = torch.multinomial(action_vec, 1).item()

        if cur_state[action_index] == 0 and cur_forbidden[action_index] != 1:
            # action
            cur_state[action_index] = 1
            action_taken[action_index] = 1
            point_added = True

            point_row = action_index//n
            point_col = action_index%n
            # fill forbidden
            for corner in corners:
                corner_row = corner//n
                corner_col = corner%n
                if corner_row < point_row and point_col < corner_col:
                    # fill left block
                    for forbidden_row in range(corner_row+1, point_row):
                        for forbidden_col in range(point_col):
                            forbidden_index = forbidden_row*n + forbidden_col
                            if cur_state[forbidden_index] == 0:
                                cur_forbidden[forbidden_index] = 1
                    # fill right block
                    for forbidden_col in range(point_col+1, corner_col):
                        for forbidden_row in range(point_row+1, n):
                            forbidden_index = forbidden_row*n + forbidden_col
                            if cur_state[forbidden_index] == 0:
                                cur_forbidden[forbidden_index] = 1
        else:
            action_vec[action_index] = 0
            action_vec = action_vec / torch.sum(action_vec)

    return cur_state, action_taken, cur_forbidden

# Defining a function to generate a new session
def generate_session(agent, corner_agent, n_sessions, n):

    # (nth session, always 4*n-4+1 steps, always n*n board)
    states = torch.zeros((n_sessions, 4*n-4+1, n*n), dype=torch.float)
    actions = torch.zeros((n_sessions, 4*n-4+1, n*n), dype=torch.float)
    forbidden_points = torch.zeros((n_sessions, 4*n-4+1, n*n), dype=torch.float)
    corners = torch.zeros((n_sessions, 4*n-4+1, n*n), dype=torch.float)

    total_score = torch.zeros([n_sessions], dtype = torch.float)


    for i in range(n_sessions):
        step = 0

        # written with 1-indexing - (1,1), (1,2), (n,1), (n-1,n), (n,n) are always included
        # these are indices 0, 1, n^2-n, n^2-n-1, n^2-1. We do not initially add (1,2) nor (n-1,n)
        # because we want to preserve the possiblity of adding them as corners.
        
        # add free points
        while step < 3:
            step += 1
            cur_state = states[i,step-1, :]

            new_state, action = add_free_points(cur_state, n, step)
            
            actions[i,step-1, :] = action
            states[i,step, :] = new_state

        # add corners
        corner_num = 0
        corner_list = [0]*(n-2) # there can be at most n-2 corners

        row_boundary, col_boundary = 0, 1

        row_zero_set = False
        terminal = False
        while not terminal:
            step += 1
            cur_state = states[i,step-1, :]

            output = corner_agent(cur_state)

            new_state, action, terminal, row_added, col_added = add_corner(cur_state, output, n, row_boundary, col_boundary)
            
            # Ensures that a corner is always set in first row. If the lower block is entered, essentially sets (1,2), 1-indexed, as the first corner
            # then artifically using the above point as the next step. The lower black can only be entered once.
            if row_added == 0:
                row_zero_set = True
            if not row_zero_set:
                cur_state = torch.clone(states[i,step-1, :])
                cur_state[1] = 1

                actions[i,step-1, 1] = 1
                states[i,step, :] = cur_state
                corners[i,step, 1] = 1
            
                corner_list[corner_num] = 1 # the first corner is at index 1
                corner_num += 1
            
                step += 1
                row_zero_set = True

                new_state[1] = 1
            ''' need to account for attempted corners in second column, other than index 1 '''
            corner_index = row_added*n + col_added
            corner_list[corner_num] = corner_index
            corners[i,step, corner_index] = 1

            corner_num += 1

            row_boundary = row_added + 1
            col_boundary = col_added + 1

            actions[i,step-1, :] = action
            states[i,step, :] = new_state

        # prune corner list
        corner_list = corner_list[:corner_num]
        corner_list.sort()

        # add induced corners
        for corner_index in corner_list:
            step += 1
            cur_state = torch.clone(states[i,step-1, :])

            corner_row = corner_index//n
            corner_col = corner_index%n
            # induced corner = (i+1, j-1) -> (i+1)n + (j-1), where i=row and j=col
            induced_corner_index = (corner_row+1)*n + (corner_col-1)
            cur_state[induced_corner_index] = 1

            actions[i,step-1, induced_corner_index] = 1
            states[i,step, :] = cur_state

        # add zig-zag and upper forbidden points
        corner_list.append(n**2-1) # n^2-1 is not a corner, but we append it to work with while-loop below
        #corner_list.append(float('inf'))
        cur_path_element = 1 # first element on the zig-zag path, besides upper-left corner
        target_corner_num = 0
        target_corner = corner_list[target_corner_num]

        # first row
        while cur_path_element != target_corner:
            # action
            step += 1
            cur_state = torch.clone(states[i,step-1, :])
            cur_state[cur_path_element] = 1

            actions[i,step-1, cur_path_element] = 1
            states[i,step, :] = cur_state

            cur_path_element += 1

        # update forbidden states
        cur_forbidden_state = torch.clone(forbidden_points[i,step-1, :])
        for forbidden_index in range(cur_path_element+1, n):
            cur_forbidden_state[forbidden_index] = 1 
        forbidden_points[i, step, :] = cur_forbidden_state

        target_corner_num += 1
        while target_corner_num < len(corner_list):
            target_corner = corner_list[target_corner_num]  
            target_row = target_corner//n
            #target_col = target_corner%n

            cur_row = cur_path_element//n
            cur_col = cur_path_element%n

            while cur_path_element != target_corner and cur_path_element != n**2-n-1: # incorrect row OR incorrect column
                if cur_row != target_row: # must move row
                    step += 1
                    # update forbidden states
                    cur_forbidden_state = torch.clone(forbidden_points[i,step-1, :])
                    for forbidden_index in range(cur_path_element+1, cur_row*n + n):
                        cur_forbidden_state[forbidden_index] = 1 
                    forbidden_points[i, step, :] = cur_forbidden_state
                    # action
                    cur_state = torch.clone(states[i,step-1, :])
                    cur_path_element += n # step element by one row, fixing column
                    cur_state[cur_path_element] = 1

                    actions[i,step-1, cur_path_element] = 1
                    states[i,step, :] = cur_state

                    cur_row += 1
                else: # cur_col != target_col, i.e, correct row but must move column
                    cur_path_element += 1
                    if cur_path_element != target_corner:
                        # action
                        step += 1

                        cur_forbidden_state = torch.clone(forbidden_points[i,step-1, :])
                        forbidden_points[i, step, :] = cur_forbidden_state

                        cur_state = torch.clone(states[i,step-1, :])
                        #cur_path_element += 1 # step element by one column, fixing row
                        cur_state[cur_path_element] = 1

                        actions[i,step-1, cur_path_element] = 1
                        states[i,step, :] = cur_state

                        cur_col += 1

            target_corner_num += 1

        # add remaining points
        terminal = False
        while step < 4*n - 4:
            step+=1
            cur_state = states[i,step-1, :]
            cur_forbidden = forbidden_points[i,step-1, :]

            #tic = time.time()
            #output = agent(cur_state)
            #pred_time += time.time() - tic

            output = agent(cur_state)

            #tic = time.time()
            new_state, action, new_forbidden_state = add_point(cur_state, output, cur_forbidden, corner_list, n)
            #play_time += time.time() - tic

            #tic = time.time()
            if terminal:
                total_score[i] = cur_state.sum()
                continue
            actions[i,step-1, :] = action
            #scoreUpdate_time += time.time() - tic

            #tic = time.time()
            states[i,step, :] = new_state
            #recordsess_time += time.time() - tic

            forbidden_points[i,step, :] = new_forbidden_state
        
        cur_state = states[i,step, :]
        mtx = cur_state.reshape(n,n)

        total_score[i] = glynn(mtx.numpy())
    return


def train(n):

    acceptence_threshold = np.inf
    # explore_rate = 0.6

    input_space = n*n
    INF = 1000000

    first_layer_neurons = 128
    second_layer_neurons = 64
    third_layer_neurons = 4
    last_layer_neurons = n*n

    # Defining the neural network architecture
    class MyNet(nn.Module):
        def __init__(self):
            super(MyNet, self).__init__()
            self.fc1 = nn.Linear(input_space, first_layer_neurons)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(first_layer_neurons, second_layer_neurons)
            self.relu = nn.ReLU()
            self.fc3 = nn.Linear(second_layer_neurons, third_layer_neurons)
            self.relu = nn.ReLU()
            self.fc4 = nn.Linear(third_layer_neurons, last_layer_neurons)
            self.softmax = nn.Softmax(dim=0)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            x = self.softmax(self.fc4(x))

            return x

    # Create neural network for corners
    corner_net = MyNet()

    # Create neural network for all other points
    net = MyNet()

    # Definte the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    corner_optimizer = optim.SGD(corner_net.parameters(), lr=Learning_rate)
    optimizer = optim.SGD(net.parameters(), lr=Learning_rate)

    ''' don't understand yet '''
    global super_states
    super_states = torch.empty((0, n*n, n*n), dtype=torch.int)
    global super_actions
    super_actions = torch.tensor([], dtype=torch.int)
    global super_rewards
    super_rewards = torch.tensor([])
    sessgen_time = 0
    fit_time = 0
    score_time = 0

    counter = 0
    pass_threshold = 1.25 * n

    cur_best_reward = 0
    cur_best_board = torch.zeros([n*n])
    cur_best_game = torch.zeros([n*n, n*n])
    cur_best_actions = torch.zeros([n*n, n*n])

    for i in range(10):

        ####
        # setup sessions
        elite_states.to(device)
        corner_outputs = corner_net(elite_states)
        outputs = net(elite_states)

        loss = criterion(corner_outputs, elite_actions.float())
        loss.backward()
        corner_optimizer.step()

        loss = criterion(outputs, elite_actions.float())
        loss.backward()
        optimizer.step()

    return net, net_corner