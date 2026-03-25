import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from helper_functions import (
    generate_session,
    save_experiment_header,
    save_best_timeline,
    save_board,
    save_reward,
    select_elites,
    select_super_sessions,
)


def main(args):
    matrix_size = args.size * args.size

    first_layer_neurons = 128
    second_layer_neurons = 64
    third_layer_neurons = 4
    last_layer_neurons = matrix_size

    # Define the neural network architecture used
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(matrix_size, first_layer_neurons)
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
        
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

    global super_states
    super_states = torch.empty((0, matrix_size, matrix_size), dtype=torch.int)
    global super_actions
    super_actions = torch.tensor([], dtype=torch.int)
    global super_rewards
    super_rewards = torch.tensor([])

    cur_best_reward = 0
    cur_best_board = torch.zeros([matrix_size])
    #cur_best_game = torch.zeros([matrix_size, matrix_size])
    #cur_best_actions = torch.zeros([matrix_size, matrix_size])

    for i in range(args.epochs):
        states_batch, actions_batch, rewards_batch = generate_session(net, args)

        if i > 0:
            states_batch = torch.cat((states_batch, super_states), dim=0)
            actions_batch = torch.cat((actions_batch, super_actions), dim=0)
            rewards_batch = torch.cat((rewards_batch, super_rewards), dim=0)

        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, args)

        super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, args)
        super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
        super_sessions.sort(key=lambda x: x[2], reverse=True)

        optimizer.zero_grad()
        elite_states.to(args.device)
        outputs = net(elite_states)

        loss = criterion(outputs, elite_actions.float())

        loss.backward()
        optimizer.step()

        super_states = torch.stack([super_sessions[i][0] for i in range(len(super_sessions))])
        super_actions = torch.stack([super_sessions[i][1] for i in range(len(super_sessions))])
        super_rewards = torch.stack([super_sessions[i][2] for i in range(len(super_sessions))])

        print("\n" + str(i+1) +  ". Best individuals: " + str(super_rewards))

        if i == 0:
            save_experiment_header(args)

        max_index = 0
        if super_rewards[max_index] > cur_best_reward:
            cur_best_reward = super_rewards[max_index]
            print('new best: ' + str(cur_best_reward))
            cur_best_board = super_states[max_index, 4 * args.size - 4].numpy()
            #cur_best_game = super_states[max_index]
            #cur_best_actions = super_actions[max_index]

            best_states_set = set()
            best_states_set.add(str(cur_best_board))
            
            #save_board(cur_best_board, args)
            #save_reward(cur_best_reward, args)
            save_best_timeline(cur_best_board, cur_best_reward, args)
                    
        if super_rewards[max_index] == cur_best_reward:
            cur_best_board = super_states[max_index, 4 * args.size - 4].numpy()
            if str(cur_best_board) not in best_states_set:
                best_states_set.add(str(cur_best_board))
                #save_board(cur_best_board, args)
                #save_reward(cur_best_reward, args)
                save_best_timeline(cur_best_board, cur_best_reward, args)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Parameter settings for training')

    # Add arguments
    parser.add_argument('--size', type=int, default='5', help='size of matrices')
    parser.add_argument('--batch_size', type=int, default='100', help='batch size')
    parser.add_argument('--percentile', type=int, default='90', help='top 100-x percentile the agent will learn from')
    parser.add_argument('--super_percentile', type=int, default='95', help='top 100-x percentile that survives to the next generation')
 
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs')
    parser.add_argument('--device', type=str, default='mps', choices=['cuda', 'cpu', 'mps'], help='device to trian on: cuda, cpu, or mps')
    parser.add_argument('--experiment_name', type=str, default='experiment', help='experiment_name')
    parser.add_argument('--data_directory', type=str, default='data', help='the directory data should be saved to')

    # Parse the arguments
    args = parser.parse_args()

    main(args)