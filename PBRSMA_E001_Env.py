from PBRSMA_E001_Agent import Agent
class MultiAgentEnv(object):
    def __init__(self,PBSR):
        #The potential list of each agent
        self.potential_list = [[0, 1, 2, 3, 2, 1, 2, 3, 4, 3, -2, 3, 4, 5, -2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8],
                               [4, 5, 6, 7, 8, 3, 4, 5, 6, 7, -2, 3, 4, 5, -2, 1, 2, 3, 4, 3, 0, 1, 2, 3, 2]]
        # self.potential_list = [[8, 7, 4, 5, 4, 2, 7, 3, 4, 3, -2, 3, 4, 5, -2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8],
        #                        [0, 1, 2, 3, 2, 1, 2, 3, 4, 3, -2, 3, 4, 5, -2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8]]
        self.num_actions = 5
        self.num_of_episodes = 100
        self.max_timesteps = 150
        self.action_label = ['west', 'east', 'north', 'south', 'stay']
        self.x_dimension = 5
        self.y_dimension = 5
        self.num_states = self.x_dimension * self.y_dimension
        self.foul_flags = [False, False] # No one fouls at the beginning
        self.PBSR=PBSR
        self.goal_reward = 10.0
        self.step_penalty = -1.0
        self.foul_penalty = -2.0
        # All the postion here is written as the number of state.
        self.obstacles = [10, 14]
        self.agent_start_states = [0, 20]
        self.goal_state = [24,4]
        self.moves_to_goal = [[], []]
        self.steps_to_goal=[0,0]
        self.alpha = 0.6
        self.gamma = 0.95
        self.epsilon = 0.8


    def setup_agent(self):
        return Agent(self.num_states, self.num_actions,
                     self.alpha, self.gamma, self.epsilon)

    def do_experiment(self):
        self.agents=[self.setup_agent(),self.setup_agent()]
        for i in range(0, self.num_of_episodes):
            self.do_episode_multi_agent_v()
            i += 1

    def reset_reach_flag(self):
        self.goal_reached_flags = [False,False]

    def initialize_agent_position(self):
        self.current_agent_states = [0,20]

    def do_episode_multi_agent_v(self):
        #initialize
        self.reset_reach_flag()
        self.initialize_agent_position()
        self.steps_to_goal = [0,0]
        #-------------------
        for i in range(0, self.max_timesteps):
            if self.goal_reached_flags[0] and self.goal_reached_flags[1]:#If the goals were reached
                break
            else:
                self.do_timestep(self.goal_reached_flags)
        self.moves_to_goal[0].append(self.steps_to_goal[0])
        self.moves_to_goal[1].append(self.steps_to_goal[1])

    def do_timestep(self,goal_reached_flags):
        #Get the index list of the agent not reached the goal yet.
        go_on_list=[i for i,x in enumerate(goal_reached_flags) if x==False]
        #The agent on their way will do the following loop
        for agent_index in go_on_list:
            self.steps_to_goal[agent_index]+=1
            #Get the other agents index
            other_agent_index_list = [o for o, y in enumerate(go_on_list) if o != agent_index]
            current_state = self.current_agent_states[agent_index]
            # This is where it went wrong !
            # if self.PBSR:
            #     selected_action = self.agents[agent_index].select_action_PBRS(current_state, self.potential_dict_s_a,
            #                                                                   self.potential_list[agent_index],
            #                                                                   agent_index)
            # else:
            selected_action = self.agents[agent_index].select_action(current_state)
            previous_state = current_state
            #The new current state is based on the action and previous state
            current_state = self.get_next_state(previous_state,selected_action,other_agent_index_list,agent_index)
            #Calculate the reward
            reward = self.calculate_reward(previous_state,selected_action,current_state,agent_index)
            self.current_agent_states[agent_index] = current_state
            #Update the q value
            self.agents[agent_index].update_q_value(previous_state, selected_action, current_state, reward)
    def get_next_state(self,previous_state,action,other_agent_index,current_agent_index):
        next_state = -1
        #Get the states occupied, including other agents position and the position of the obstacles
        occupied_states = [self.current_agent_states[i] for i in other_agent_index]+self.obstacles
        if action == 0 : #west
            if previous_state-5 in occupied_states or previous_state-5 < 0:#limit the agent away from the occupied states and border
                self.foul_flags[current_agent_index] = True
                next_state = previous_state
            else:
                next_state = previous_state-5
        if action == 1:#east
            if previous_state+5 in occupied_states or previous_state+5 > 24:#limit the agent away from the occupied states and border
                self.foul_flags[current_agent_index] = True
                next_state = previous_state
            else:
                next_state = previous_state+5
        if action == 2:#north
            if previous_state+1 in occupied_states+[5,10,20,25]:#limit the agent away from the occupied states and border
                self.foul_flags[current_agent_index] = True
                next_state = previous_state
            else:
                next_state = previous_state+1
        if action == 3:#south
            if previous_state-1 in occupied_states+[-1,4,14,19]:#limit the agent away from the occupied states and border
                self.foul_flags[current_agent_index] = True
                next_state = previous_state
            else:
                next_state = previous_state-1
        elif action == 4:#stay
            next_state = previous_state
        return next_state

    def calculate_reward(self,previous_state_num,selected_action,current_state_num,agent_index):
        reward = 0.0
        pure_reward=0.0
        if current_state_num == self.goal_state[agent_index] and not self.goal_reached_flags[agent_index]:
            self.goal_reached_flags[agent_index] = True
            reward = self.goal_reward
            pure_reward = reward
        elif self.foul_flags[agent_index]:
            reward = self.foul_penalty
            pure_reward = reward
        else:
            reward = self.step_penalty
            pure_reward = reward
        #This make it right
        if self.PBSR:#If PBSR take the potential list into consideration
            reward += self.gamma*self.potential_list[agent_index][current_state_num]-self.potential_list[agent_index][previous_state_num]
        return (reward,pure_reward)





