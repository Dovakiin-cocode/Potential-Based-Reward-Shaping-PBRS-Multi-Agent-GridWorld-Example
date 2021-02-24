import numpy as np
import pandas as pd
import random

class Agent(object):
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states  # The total number of states
        self.num_actions = num_actions  # The total number of actions
        self.q_table = pd.DataFrame(np.zeros((self.num_states, self.num_actions)))
        self.q_table_pure = pd.DataFrame(np.zeros((self.num_states, self.num_actions)))
        self.alpha = alpha  # learning rate alpha
        self.gamma = gamma  # discounting rate gamma
        self.epsilon = epsilon  # epsilon is the param balancing exploration and exploitation

    def update_q_value(self, previous_state, selected_action, current_state, reward):
        old_Q = self.q_table.iloc[previous_state].at[selected_action]
        max_Q = self.q_table.iloc[current_state].max()
        new_Q = old_Q + self.alpha * (reward[0] + self.gamma * max_Q - old_Q)
        new_Q_pure = old_Q + self.alpha * (reward[1] + self.gamma * max_Q - old_Q)
        self.q_table.iloc[previous_state].at[selected_action] = new_Q
        # print("state",previous_state," ",current_state)
        self.q_table_pure.iloc[previous_state].at[selected_action] = new_Q_pure

    def select_action(self, state):
        if np.random.uniform() > self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:  
            s = self.q_table.iloc[state]
            action = s[s == s.max()].index 
            action = np.random.choice(action)
        return action

    # This is the method which makes the result not beautiful
    # def select_action_PBRS(self, state, potential_dict, potential_list,agent_index):
    #     if np.random.uniform() > self.epsilon:
    #         # If so choose a random action
    #         action = random.randint(0, self.num_actions - 1)
    #     else:  # else choose the current max value of this state from Q table
    #         s = self.q_table.iloc[state]
    #         max_value = 0
    #         action = 0
    #         total_actions_pair = []
    #         for i in range(0,5):
    #             potential_cur = potential_list[potential_dict[state][i]]
    #             total = self.gamma*potential_cur + self.q_table[i][state] - potential_list[state]
    #             total_actions_pair.append((i,total))
    #         sort_pair=sorted(total_actions_pair,key=lambda x:x[1])
    #         sort_sliced_pair=sort_pair[3:5]
    #         r=random.randint(0,1)
    #         action = sort_sliced_pair[r][0]
    #     return action
    #
    # def print_q_table(self):
    #     print(self.q_table)