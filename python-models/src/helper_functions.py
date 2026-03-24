import numpy as np
import torch

from permanents import glynn


def generate_session(agent, args):# n_sessions, n, args):
    # (nth session, always 4*n - 4 steps, always n*n board)
    states = torch.zeros((args.batch_size, (4 * args.size - 4) + 1, args.size * args.size))
    actions = torch.zeros((args.batch_size, (4 * args.size - 4) + 1, args.size * args.size))
    scores = torch.zeros([args.batch_size])

    states.to(args.device)

    for i in range(args.batch_size):
        step = 0

        while step < 4 * args.size - 4:
            step+=1
            cur_state = states[i, step-1, :]

            output = agent(cur_state)

            new_state, action = add_point(cur_state, output, args.size)

            actions[i, step-1, :] = action
            states[i, step, :] = new_state

        final_state = states[i,step, :]
        state_mtx = final_state.reshape(args.size, args.size)
        scores[i] = glynn(state_mtx.numpy())

    return states, actions, scores


def select_super_sessions(n_sessions, states_batch, actions_batch, rewards_batch, percentile=90):
    counter = n_sessions * (100 - percentile)/100
    reward_threshold = np.percentile(rewards_batch, percentile)

    super_states = torch.empty(0)
    super_actions = torch.empty(0)
    super_rewards = torch.empty(0)

    for i in range(len(states_batch)):
        if counter <= 0:
            break

        if rewards_batch[i] >= reward_threshold - 0.001:
            super_states = torch.cat((super_states, states_batch[i].unsqueeze(0)), dim=0)
            super_actions = torch.cat((super_actions, actions_batch[i].unsqueeze(0)), dim=0)
            super_rewards = torch.cat((super_rewards, torch.tensor([rewards_batch[i]])), dim=0)
            counter -= 1

    return super_states, super_actions, super_rewards


def select_elites(n_sessions, states_batch, actions_batch, rewards_batch, percentile=50):
    counter = n_sessions * (100 - percentile)/100
    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_states = torch.empty(0)
    elite_actions = torch.empty(0)

    for i in range(len(states_batch)):
        if counter <= 0:
            break

        if rewards_batch[i] >= reward_threshold - 0.01:
            game_end_index = 0
            for item in states_batch[i]:
                if item.sum() == 0 and game_end_index != 0:
                    break
                elite_states = torch.cat((elite_states, item.unsqueeze(0)))
                game_end_index += 1

            for item in actions_batch[i]:
                if game_end_index == 0:
                    break
                elite_actions = torch.cat((elite_actions, item.unsqueeze(0)))
                game_end_index -= 1
            counter -= 1

    return elite_states, elite_actions


def new_point_allowed(one_indices, new_point_index, n):
    row = new_point_index//n
    col = new_point_index%n
    point_allowed = True

    for i in range(len(one_indices)):
        for j in range(i+1, len(one_indices)):
            point_one_row = one_indices[i] // n
            point_one_col = one_indices[i] % n
            point_two_row = one_indices[j] // n
            point_two_col = one_indices[j] % n

            if (   row == point_one_row
                or row == point_two_row
                or point_one_row == point_two_row
                or col == point_one_col
                or col == point_two_col
                or point_one_col == point_two_col
            ):
                continue
            
            # ensure point_one_col < point_two_col
            if point_two_col < point_one_col:
                point_one_row, point_two_row = point_two_row, point_one_row
                point_one_col, point_two_col = point_two_col, point_one_col

            # new point as 1 in a valid 312-pattern
            if (point_two_row < point_one_row < row and point_one_col < col < point_two_col):
                point_allowed = False
                break

            # new point as 2 in a valid 312-pattern
            if (point_two_row < row < point_one_row and col < point_one_col):
                point_allowed = False
                break

            # new point as 3 in a valid 312-pattern
            if (row < point_one_row < point_two_row and point_two_col < col):
                point_allowed = False
                break

        if point_allowed == False:
            break

    return point_allowed


# defines a helper function to add point, action_vec is the output, output = agent(cur_state), agent = n
def add_point(input_state, action_vec, n):

    point_added = False
    action_taken = torch.zeros([len(action_vec)])
    cur_state = torch.clone(input_state)

    while not point_added:
        action_index = torch.multinomial(action_vec, 1).item()
        
        one_indices = torch.flatten(torch.nonzero(cur_state))
        action_allowed = new_point_allowed(one_indices, action_index, n)

        if cur_state[action_index] == 0 and action_allowed:
            cur_state[action_index] = 1
            action_taken[action_index] = 1
            point_added = True
        else:
            action_vec[action_index] = 0
            action_vec = action_vec / torch.sum(action_vec)

    return cur_state, action_taken


# helper function to add 3 of the 5 free points present in all 312-avoiding matrices
def add_free_points(input_state, n, step):
    action_taken = torch.zeros([len(input_state)])
    cur_state = torch.clone(input_state)

    if step == 1:
        action_index = 0
    elif step == 2:
        action_index = n**2-n
    else: # step == 3:
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

        if (   (action_row == n-2 and action_col != n-1) 
            or (action_col == 1 and action_row != 0) 
            or (row_boundary == 0 and action_row != 0)
        ):
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
def add_point_and_forbidden_state(input_state, action_vec, forbidden_state, corners, n):
    
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
                # fill left block
                if corner_row < point_row and point_col < corner_col:
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

