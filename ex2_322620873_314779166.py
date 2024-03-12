ids = ["322620873", "314779166"]

RESET_PENALTY = -2
DROP_IN_DESTINATION_REWARD = 4
MARINE_COLLISION_PENALTY = -1
GAMMA = 0.9

from itertools import product
import random

class OptimalPirateAgent:
    def __init__(self, initial):
        self.map = initial['map']
        self.time_step = initial['turns to go']
        base_row = [x for x in self.map if 'B' in x][0]
        self.base = (self.map.index(base_row), base_row.index('B'))
        state_dict = dict()

        possible_locations = list()
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                if self.map[i][j] != 'I':
                    possible_locations.append((i, j))
        pirates_locations_and_capacities = {pirate: [(pirate, location, capacity) for capacity in range(3) for location in possible_locations] for pirate in initial['pirate_ships'].keys()}
        state_dict['pirates'] = list(product(*pirates_locations_and_capacities.values()))  # A cartesian product, in order to get all possible combinations of pirates' locations.

        treasures_info = {treasure: [(treasure, location) for location in initial['treasures'][treasure]['possible_locations']] for treasure in initial['treasures'].keys()}
        state_dict['treasures'] = list(product(*treasures_info.values()))  # A cartesian product, in order to get all possible combinations of treasures' locations.

        marines_info = {marine: [(marine, index) for index in range(len(initial['marine_ships'][marine]['path']))] for marine in initial['marine_ships'].keys()}
        state_dict['marines'] = list(product(*marines_info.values()))  # A cartesian product, in order to get all possible combinations of marines' locations.

        all_states = list(product(*state_dict.values()))  # A cartesian product, in order to get all possible combinations of states.
        self.next_states = dict()

        self.value_iterations = dict() #{state: [(0, None)]*(initial['turns to go']+1) for state in all_states} # A dictionary of all possible states, with a list of 100 tuples, each tuple contains the action that maximizes the value and the value of the state.
        # self.value_iterations[state] = (value of the state, action that maximizes the value)
        
        self.initial_state = self.create_state(initial)

        self.len_rows = len(self.map)
        self.len_cols = len(self.map[0])
        for t in range(0, 1+initial['turns to go']):
            for state in all_states:
                if t == 0:
                    self.value_iterations[state] = [(0, None)]*(1+initial['turns to go'])
                else:
                    self.updating_value(state, t)

    def is_valid_location(self, x, y):
        return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (self.map[x][y] != 'I')
    
    def is_valid_island(self, x, y):
        return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (self.map[x][y] == 'I')
    
    def create_state(self, initial):
        pirates_info = tuple()
        for pirate in initial['pirate_ships'].keys():
            pirates_info += ((pirate, initial['pirate_ships'][pirate]['location'], initial['pirate_ships'][pirate]['capacity']),)

        self.treasures = {treasure: dict() for treasure in initial['treasures']}
        treasures_locations = tuple()
        for treasure in initial['treasures'].keys():
            possible_locations = initial['treasures'][treasure]['possible_locations']
            self.treasures[treasure]['possible_locations'] = possible_locations
            self.treasures[treasure]['prob_change_location'] = initial['treasures'][treasure]['prob_change_location']/len(possible_locations)
            treasures_locations += ((treasure, initial['treasures'][treasure]['location']),)
        
        marines_locations = tuple()
        self.marines_paths = dict()
        for marine in initial['marine_ships'].keys():
            self.marines_paths[marine] = initial['marine_ships'][marine]['path']
            marines_locations += ((marine, initial['marine_ships'][marine]['index']),)
        
        return (pirates_info, treasures_locations, marines_locations)


    def actions(self, state):
        pirates_actions = dict()
        pirates_info = state[0]
        all_possible_actions = list()

        all_possible_actions.append('terminate')
        all_possible_actions.append('reset')

        for pirate in pirates_info:
            pirates_actions[pirate] = list()
            pirates_actions[pirate].append(('wait', pirate[0]))

            for index_change in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = pirate[1][0] + index_change[0]
                new_y = pirate[1][1] + index_change[1]
                if self.is_valid_location(new_x, new_y):
                    pirates_actions[pirate].append(('sail', pirate[0], (new_x, new_y)))

                if self.is_valid_island(new_x, new_y) and pirate[2] > 0:
                    for treasure in state[1]:
                        if (new_x, new_y) == treasure[1]:
                            pirates_actions[pirate].append(('collect', pirate[0], treasure[0]))
            if (pirate[1] == self.base) and (pirate[2] < 2):
                pirates_actions[pirate].append(('deposit', pirate[0]))

        all_possible_actions += list(product(*pirates_actions.values()))
    
        return all_possible_actions
            
    def find_next_states(self, state, action):
        next_states_list = list()
        pirates_info = tuple()

        if action == 'reset':
            prob = 1  # Probability of reacing this state given this action.
            new_state_reward = RESET_PENALTY
            next_states_list.append((self.initial_state, prob, new_state_reward)) # New state: (pirates_info, self.initial_state[1], self.initial_state[2])
        
        elif action == 'terminate':
            prob = 1
            next_states_list.append((None, prob, new_state_reward))  # New state: None (game is over, no next state).
        
        else:
            marines_info = {marine[0]: list() for marine in state[2]}
            for marine in state[2]:
                if len(self.marines_paths[marine[0]]) == 1:
                    marines_info[marine[0]].append((marine[0], marine[1], 1))
                elif marine[1] == 0:
                    marines_info[marine[0]].append((marine[0], marine[1], 1/2))
                    marines_info[marine[0]].append((marine[0], marine[1]+1, 1/2))
                elif marine[1] == len(self.marines_paths[marine[0]]) - 1:
                    marines_info[marine[0]].append((marine[0], marine[1], 1/2))
                    marines_info[marine[0]].append((marine[0], marine[1]-1, 1/2))
                else:
                    marines_info[marine[0]].append((marine[0], marine[1], 1/3))
                    marines_info[marine[0]].append((marine[0], marine[1]-1, 1/3))
                    marines_info[marine[0]].append((marine[0], marine[1]+1, 1/3))
            all_marines_possible_locations_combinations = list(product(*marines_info.values()))  # A cartesian product, in order to get all possible combinations of marines' locations.

            treasures_info = {treasure[0]: list() for treasure in state[1]}
            for treasure in state[1]:
                for possible_location in self.treasures[treasure[0]]['possible_locations']:
                    prob = self.treasures[treasure[0]]['prob_change_location']
                    if possible_location == treasure[1]:
                        prob += (1 - prob*len(self.treasures[treasure[0]]['possible_locations']))
                    treasures_info[treasure[0]].append((treasure[0], possible_location, prob))
            all_treasures_possible_locations_combinations = list(product(*treasures_info.values()))  # A cartesian product, in order to get all possible combinations of treasures' locations.

            # TODO: Implement pirates rewards, cartesian product of all possible marines and treasures combinations actions.
            # multiply probabilities of marines and treasures combination. p*value(state)- value of the reward from the action if marine encounters a pirates
            
            for marines_action in all_marines_possible_locations_combinations:
                for treasures_action in all_treasures_possible_locations_combinations:
                    new_state_reward = 0
                    prob = 1
                    pirates_info = tuple()
                    treasures_locations = tuple()
                    marines_locations = tuple()
            
                    for pirate, pirate_action in zip(state[0], action):
                        capacity = pirate[2]
                        pirate_next_location = pirate[1]  # If action is not 'sail', pirate_next_location doesn't change.
                        if pirate_action[0] == 'sail':
                            pirate_next_location = pirate_action[2]

                        if pirate_action[0] == 'deposit':
                            new_state_reward += DROP_IN_DESTINATION_REWARD * (2 - capacity)
                            capacity = 2

                        caught_by_marine = False
                        for marine_action in marines_action:
                            marine_loc = self.marines_paths[marine_action[0]][marine_action[1]]
                            prob *= marine_action[2]
                            marines_locations += ((marine_action[0], marine_action[1]),)
                            
                            if (pirate_action[0] != 'deposit') and (pirate_next_location == marine_loc):
                                caught_by_marine = True
                                    
                        if (not caught_by_marine) and (pirate_action[0] == 'collect'):
                            capacity -= 1

                        if caught_by_marine:
                            new_state_reward += MARINE_COLLISION_PENALTY
                            capacity = 2

                        pirates_info += ((pirate[0], pirate_next_location, capacity),)
                        
                    for treasure_action in treasures_action:
                        prob *= treasure_action[2]
                        treasures_locations += ((treasure_action[0], treasure_action[1]),)

                    next_states_list.append(((pirates_info, treasures_locations, marines_locations), prob, new_state_reward))
        self.next_states[(state, action)] = next_states_list


    def updating_value(self, state, t):
        max_value_and_action = (float('-inf'), None)
        for action in self.actions(state): # all possible actions of the state
            if action == 'terminate':
                max_value_and_action = (0, 'terminate') #[(self.value_iterations[state][t-1][0], 'terminate')]
                continue
            if (state, action) not in self.next_states:
                self.find_next_states(state, action)

            next_states_list = self.next_states[(state, action)]
            value_of_action = sum(next_state[1] * (next_state[2] + self.value_iterations[next_state[0]][t-1][0]) for next_state in next_states_list)

            if value_of_action >= max_value_and_action[0]:
                max_value_and_action = (value_of_action, action)
        self.value_iterations[state][t] = max_value_and_action  


    def act(self, state):
        state = self.create_state(state)
        action = self.value_iterations[state][self.time_step][1]
        print(self.time_step)
        self.time_step -= 1
        print(state)
        return action

class PirateAgent:
    def __init__(self, initial):
        pass

    def act(self, state):
        pass

class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma
        self.map = initial['map']
        self.epsilon = 10**(-2)
        base_row = [x for x in self.map if 'B' in x][0]
        self.base = (self.map.index(base_row), base_row.index('B'))
        state_dict = dict()
        self.terminated = False


        possible_locations = list()
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                if self.map[i][j] != 'I':
                    possible_locations.append((i, j))
        pirates_locations_and_capacities = {pirate: [(pirate, location, capacity) for capacity in range(3) for location in possible_locations] for pirate in initial['pirate_ships'].keys()}
        state_dict['pirates'] = list(product(*pirates_locations_and_capacities.values()))  # A cartesian product, in order to get all possible combinations of pirates' locations.

        treasures_info = {treasure: [(treasure, location) for location in initial['treasures'][treasure]['possible_locations']] for treasure in initial['treasures'].keys()}
        state_dict['treasures'] = list(product(*treasures_info.values()))  # A cartesian product, in order to get all possible combinations of treasures' locations.

        marines_info = {marine: [(marine, index) for index in range(len(initial['marine_ships'][marine]['path']))] for marine in initial['marine_ships'].keys()}
        state_dict['marines'] = list(product(*marines_info.values()))  # A cartesian product, in order to get all possible combinations of marines' locations.

        all_states = list(product(*state_dict.values()))  # A cartesian product, in order to get all possible combinations of states.
        self.next_states = dict()

        self.value_iterations = dict() # {state: (0, None) for state in all_states} # A dictionary of all possible states, with a list of 100 tuples, each tuple contains the action that maximizes the value and the value of the state.
        # self.value_iterations[state] = (value of the state, action that maximizes the value)
        
        self.initial_state = self.create_state(initial)

        self.len_rows = len(self.map)
        self.len_cols = len(self.map[0])
        self.t = 0
        while True:
            max_state = float('-inf')
            for state in all_states:
                if self.t == 0:
                    self.value_iterations[state] = (0, None)   
                else:
                    previous_t_value_state = self.value_iterations[state][0]
                    self.updating_value(state)
                    current_t_value_state = self.value_iterations[state][0]
                    distance = abs(current_t_value_state - previous_t_value_state)
                    if distance > max_state:
                        max_state = distance
            if self.t > 0 and max_state < self.epsilon:
                break
            self.t += 1

    def is_valid_location(self, x, y):
        return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (self.map[x][y] != 'I')
    
    def is_valid_island(self, x, y):
        return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (self.map[x][y] == 'I')
    
    def create_state(self, initial):
        pirates_info = tuple()
        for pirate in initial['pirate_ships'].keys():
            pirates_info += ((pirate, initial['pirate_ships'][pirate]['location'], initial['pirate_ships'][pirate]['capacity']),)

        self.treasures = {treasure: dict() for treasure in initial['treasures']}
        treasures_locations = tuple()
        for treasure in initial['treasures'].keys():
            possible_locations = initial['treasures'][treasure]['possible_locations']
            self.treasures[treasure]['possible_locations'] = possible_locations
            self.treasures[treasure]['prob_change_location'] = initial['treasures'][treasure]['prob_change_location']/len(possible_locations)
            treasures_locations += ((treasure, initial['treasures'][treasure]['location']),)
        
        marines_locations = tuple()
        self.marines_paths = dict()
        for marine in initial['marine_ships'].keys():
            self.marines_paths[marine] = initial['marine_ships'][marine]['path']
            marines_locations += ((marine, initial['marine_ships'][marine]['index']),)
        
        return (pirates_info, treasures_locations, marines_locations)


    def actions(self, state):
        pirates_actions = dict()
        pirates_info = state[0]
        all_possible_actions = list()

        all_possible_actions.append('terminate')
        all_possible_actions.append('reset')

        for pirate in pirates_info:
            pirates_actions[pirate] = list()
            pirates_actions[pirate].append(('wait', pirate[0]))

            for index_change in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = pirate[1][0] + index_change[0]
                new_y = pirate[1][1] + index_change[1]
                if self.is_valid_location(new_x, new_y):
                    pirates_actions[pirate].append(('sail', pirate[0], (new_x, new_y)))

                if self.is_valid_island(new_x, new_y) and pirate[2] > 0:
                    for treasure in state[1]:
                        if (new_x, new_y) == treasure[1]:
                            pirates_actions[pirate].append(('collect', pirate[0], treasure[0]))
            if (pirate[1] == self.base) and (pirate[2] < 2):
                pirates_actions[pirate].append(('deposit', pirate[0]))

        all_possible_actions += list(product(*pirates_actions.values()))
        return all_possible_actions
            
            
    def find_next_states(self, state, action):
        next_states_list = list()
        pirates_info = tuple()

        if action == 'reset':
            prob = 1  # Probability of reacing this state given this action.
            new_state_reward = RESET_PENALTY
            next_states_list.append((self.initial_state, prob, new_state_reward)) # New state: (pirates_info, self.initial_state[1], self.initial_state[2])
        
        elif action == 'terminate':
            prob = 1
            next_states_list.append((None, prob, new_state_reward))  # New state: None (game is over, no next state).
        
        else:
            marines_info = {marine[0]: list() for marine in state[2]}
            for marine in state[2]:
                if len(self.marines_paths[marine[0]]) == 1:
                    marines_info[marine[0]].append((marine[0], marine[1], 1))
                elif marine[1] == 0:
                    marines_info[marine[0]].append((marine[0], marine[1], 1/2))
                    marines_info[marine[0]].append((marine[0], marine[1]+1, 1/2))
                elif marine[1] == len(self.marines_paths[marine[0]]) - 1:
                    marines_info[marine[0]].append((marine[0], marine[1], 1/2))
                    marines_info[marine[0]].append((marine[0], marine[1]-1, 1/2))
                else:
                    marines_info[marine[0]].append((marine[0], marine[1], 1/3))
                    marines_info[marine[0]].append((marine[0], marine[1]-1, 1/3))
                    marines_info[marine[0]].append((marine[0], marine[1]+1, 1/3))
            all_marines_possible_locations_combinations = list(product(*marines_info.values()))  # A cartesian product, in order to get all possible combinations of marines' locations.

            treasures_info = {treasure[0]: list() for treasure in state[1]}
            for treasure in state[1]:
                for possible_location in self.treasures[treasure[0]]['possible_locations']:
                    prob = self.treasures[treasure[0]]['prob_change_location']
                    if possible_location == treasure[1]:
                        prob += (1 - prob*len(self.treasures[treasure[0]]['possible_locations']))
                    treasures_info[treasure[0]].append((treasure[0], possible_location, prob))
            all_treasures_possible_locations_combinations = list(product(*treasures_info.values()))  # A cartesian product, in order to get all possible combinations of treasures' locations.
  
            for marines_action in all_marines_possible_locations_combinations:
                for treasures_action in all_treasures_possible_locations_combinations:
                    new_state_reward = 0
                    prob = 1
                    pirates_info = tuple()
                    treasures_locations = tuple()
                    marines_locations = tuple()
            
                    for pirate, pirate_action in zip(state[0], action):
                        capacity = pirate[2]
                        pirate_next_location = pirate[1]  # If action is not 'sail', pirate_next_location doesn't change.
                        if pirate_action[0] == 'sail':
                            pirate_next_location = pirate_action[2]

                        if pirate_action[0] == 'deposit':
                            new_state_reward += DROP_IN_DESTINATION_REWARD * (2 - capacity)
                            capacity = 2

                        caught_by_marine = False
                        for marine_action in marines_action:
                            marine_loc = self.marines_paths[marine_action[0]][marine_action[1]]
                            prob *= marine_action[2]
                            marines_locations += ((marine_action[0], marine_action[1]),)
                            
                            if (pirate_action[0] != 'deposit') and (pirate_next_location == marine_loc):
                                caught_by_marine = True
                                    
                        if (not caught_by_marine) and (pirate_action[0] == 'collect'):
                            capacity -= 1

                        if caught_by_marine:
                            new_state_reward += MARINE_COLLISION_PENALTY
                            capacity = 2

                        pirates_info += ((pirate[0], pirate_next_location, capacity),)
                        
                    for treasure_action in treasures_action:
                        prob *= treasure_action[2]
                        treasures_locations += ((treasure_action[0], treasure_action[1]),)

                    next_states_list.append(((pirates_info, treasures_locations, marines_locations), prob, new_state_reward))
        self.next_states[(state, action)] = next_states_list


    def updating_value(self, state):
        max_value_and_action = (float('-inf'), None)
        for action in self.actions(state): # all possible actions of the state
            if action == 'terminate':
                max_value_and_action = (0, 'terminate')
                continue
            if (state, action) not in self.next_states:
                self.find_next_states(state, action)

            next_states_list = self.next_states[(state, action)]
            value_of_action = self.gamma * sum(next_state[1] * (next_state[2] + self.value_iterations[next_state[0]][0]) for next_state in next_states_list)

            if value_of_action >= max_value_and_action[0]:
                max_value_and_action = (value_of_action, action)
        self.value_iterations[state] = max_value_and_action  

    def act(self, state):
        state = self.create_state(state)
        action = self.value_iterations[state][1]
        print(state)
        print(action)
        print()
        return action

    def value(self, state):
        return self.value_iterations[state][0]


