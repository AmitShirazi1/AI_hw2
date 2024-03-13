ids = ["322620873", "314779166"]

RESET_PENALTY = -2
DROP_IN_DESTINATION_REWARD = 4
MARINE_COLLISION_PENALTY = -1
GAMMA = 0.9

from itertools import product

class OptimalPirateAgent:
    def __init__(self, initial):
        self.map = initial['map']
        self.time_step = initial['turns to go']
        self.len_rows = len(self.map)
        self.len_cols = len(self.map[0])
        # Finding the base location
        base_row = [x for x in self.map if 'B' in x][0]
        self.base = (self.map.index(base_row), base_row.index('B'))
        state_dict = dict()

        # Finding all possible locations for the pirates
        possible_locations = list()
        for i in range(self.len_rows):
            for j in range(self.len_cols):
                if self.map[i][j] != 'I':
                    possible_locations.append((i, j))

        # Finding all possible states for the pirates - each one includes the location and the capacity of the pirate
        pirates_locations_and_capacities = {pirate: [(pirate, location, capacity) for capacity in range(3) for location in possible_locations] for pirate in initial['pirate_ships'].keys()}
        state_dict['pirates'] = list(product(*pirates_locations_and_capacities.values()))  # A cartesian product, in order to get all possible combinations of pirates' locations.

        # Finding all possible states for the treasures - each one includes the location of the treasure
        treasures_info, self.treasures = dict(), dict()
        for treasure in initial['treasures'].keys():
            self.treasures[treasure] = dict()
            possible_locations = initial['treasures'][treasure]['possible_locations']
            self.treasures[treasure]['possible_locations'] = possible_locations
            self.treasures[treasure]['prob_change_location'] = initial['treasures'][treasure]['prob_change_location']/len(possible_locations)  # The probability of changing the location of the treasure to any other location equals to the probability of changing location divided by the number of possible locations.
            treasures_info[treasure] = [(treasure, location) for location in initial['treasures'][treasure]['possible_locations']]
        state_dict['treasures'] = list(product(*treasures_info.values()))  # A cartesian product, in order to get all possible combinations of treasures' locations.

        marines_info, self.marines_paths = dict(), dict()
        # Finding all possible states for the marines - each one includes the index of the marine
        for marine in initial['marine_ships'].keys():
            marines_info[marine] = [(marine, index) for index in range(len(initial['marine_ships'][marine]['path']))]
            self.marines_paths[marine] = initial['marine_ships'][marine]['path']

        state_dict['marines'] = list(product(*marines_info.values()))  # A cartesian product, in order to get all possible combinations of marines' locations.

        all_states = list(product(*state_dict.values()))  # A cartesian product, in order to get all possible combinations of states.
        self.next_states = dict()  # A dictionary of all possible states and actions as keys, with a list of all possible next states and their probabilities as values.

        # A dictionary of all possible states, with a list of 100 tuples, each tuple contains the action that maximizes the value and the value of the state.
        self.value_iterations = dict() # A dictionary of the form {state: [(value of the state, action that maximizes the value)]*(initial['turns to go']+1) for state in all_states}
        
        self.initial_state = self.create_state(initial)
        # Calculating the value of each state for each time step using value iterations algorithm.
        for t in range(0, 1+initial['turns to go']):
            for state in all_states:
                # If it's the first time step, the value of each state is 0 according to the algorithm.
                if t == 0:
                    self.value_iterations[state] = [(0, None)]*(1+initial['turns to go'])
                else:
                    self.updating_value(state, t)

    def is_valid_location(self, x, y):
        """ Check if the location (x, y) is valid (and not an island). """
        return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (self.map[x][y] != 'I')
    
    def is_valid_island(self, x, y):
        """ Check if the location (x, y) is an island. """
        return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (self.map[x][y] == 'I')
    
    def create_state(self, initial):
        """ Convert a given state in a dictionary format to a tuple format, while extracting the relevant information for our implementation. """
        pirates_info = tuple()
        # Extracting the location and capacity of each pirate and adding it to the state.
        for pirate in initial['pirate_ships'].keys():
            pirates_info += ((pirate, initial['pirate_ships'][pirate]['location'], initial['pirate_ships'][pirate]['capacity']),)

        treasures_locations = tuple()
        # Extracting the location of each treasure and adding it to the state.
        for treasure in initial['treasures'].keys():
            treasures_locations += ((treasure, initial['treasures'][treasure]['location']),)
        
        marines_locations = tuple()
        for marine in initial['marine_ships'].keys():
            # Extracting the index in the path of each marine and adding it to the state.
            marines_locations += ((marine, initial['marine_ships'][marine]['index']),)
        
        return (pirates_info, treasures_locations, marines_locations)


    def actions(self, state):
        """ Find all possible actions for a given state. """
        pirates_actions = dict()
        pirates_info = state[0]
        all_possible_actions = list()

        # Adding the actions 'terminate' and 'reset' to the list of all possible actions.
        all_possible_actions.append('terminate')
        all_possible_actions.append('reset')

        for pirate in pirates_info:
            pirates_actions[pirate] = list()
            pirates_actions[pirate].append(('wait', pirate[0]))  # Adding the action 'wait' to the list of all possible actions for each pirate.

            # Adding the actions "sail to" and "collect from" an adjacent location to the list of all possible actions for each pirate.
            for index_change in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = pirate[1][0] + index_change[0]
                new_y = pirate[1][1] + index_change[1]
                if self.is_valid_location(new_x, new_y):  # Only if the new location is valid, we add the 'sail' action to the list of all possible actions.
                    pirates_actions[pirate].append(('sail', pirate[0], (new_x, new_y)))

                if self.is_valid_island(new_x, new_y) and pirate[2] > 0:  # Only if the new location is an island and the pirate has a capacity > 0, we add the 'collect' action to the list of all possible actions.
                    for treasure in state[1]:
                        if (new_x, new_y) == treasure[1]:  # Finding the name of the adjacent treasure we can collect.
                            pirates_actions[pirate].append(('collect', pirate[0], treasure[0]))
            # If the pirate is in the base and holds at least one treasure, we add the 'deposit' action to the list of all possible actions.
            if (pirate[1] == self.base) and (pirate[2] < 2):
                pirates_actions[pirate].append(('deposit', pirate[0]))

        all_possible_actions += list(product(*pirates_actions.values()))  # Adding all possible combinations of actions for all pirates to the list of all possible actions.
        return all_possible_actions


    def find_next_states(self, state, action):
        """ Find all next states we can get to from the given state and action.

            @param state: A tuple representing the current state.
            @param action: A tuple representing the action we took from the current state.
            
            @return: A list of all possible next states given this state and action,
                     where each next state is represented by a tuple of (state, probability, reward).
        """
        next_states_list = list()
        pirates_info = tuple()

        # If the action is 'reset', we add the initial state to the list of all next states.
        if action == 'reset':
            prob = 1  # Probability of reaching the initial state with the action 'reset'.
            new_state_reward = RESET_PENALTY  # Reward of making the action 'reset'.
            next_states_list.append((self.initial_state, prob, new_state_reward))
        
        # If the action is 'terminate', we add None to the list of all next states (game is over, no next state).
        elif action == 'terminate':
            prob = 1  # Probability of reaching None with the action 'terminate'.
            next_states_list.append((None, prob, new_state_reward))  # New state: None, Reward: 0.
        
        else:
            marines_info = {marine[0]: list() for marine in state[2]}  # A dictionary of all possible indices in the path and the probability of it, for each marine.
            for marine in state[2]:
                # If the marine has only one location in its path, we add it to the list of all possible indices with probability 1.
                if len(self.marines_paths[marine[0]]) == 1:
                    marines_info[marine[0]].append((marine[0], marine[1], 1))

                # If the marine is in the first location in its path, we add it to the list of all possible indices with probability 1/2, and also probability 1/2 to the next location in the path.
                elif marine[1] == 0:
                    marines_info[marine[0]].append((marine[0], marine[1], 1/2))
                    marines_info[marine[0]].append((marine[0], marine[1]+1, 1/2))
                
                # If the marine is in the last location in its path, we add it to the list of all possible indices with probability 1/2, and also probability 1/2 to the previous location in the path.
                elif marine[1] == len(self.marines_paths[marine[0]]) - 1:
                    marines_info[marine[0]].append((marine[0], marine[1], 1/2))
                    marines_info[marine[0]].append((marine[0], marine[1]-1, 1/2))

                # If the marine is in any other location in its path, we add it to the list of all possible indices with probability 1/3, and also probability 1/3 to the previous and next locations in the path.
                else:
                    marines_info[marine[0]].append((marine[0], marine[1], 1/3))
                    marines_info[marine[0]].append((marine[0], marine[1]-1, 1/3))
                    marines_info[marine[0]].append((marine[0], marine[1]+1, 1/3))
            all_marines_possible_locations_combinations = list(product(*marines_info.values()))  # A cartesian product, in order to get all possible combinations of marines' locations.

            treasures_info = {treasure[0]: list() for treasure in state[1]}
            # For each treasure, we add all possible combinations of its name, location and probability of changing location to the list of all possible combinations of treasures' locations.
            for treasure in state[1]:
                for possible_location in self.treasures[treasure[0]]['possible_locations']:
                    prob = self.treasures[treasure[0]]['prob_change_location']
                    if possible_location == treasure[1]:
                        prob += (1 - prob*len(self.treasures[treasure[0]]['possible_locations']))
                    treasures_info[treasure[0]].append((treasure[0], possible_location, prob))
            all_treasures_possible_locations_combinations = list(product(*treasures_info.values()))  # A cartesian product, in order to get all possible combinations of treasures' locations.

            # For each combination of marines' locations and treasures' locations, we add all possible combinations of pirates' locations and capacities to the list of all next states.
            # We do that because the environment step (treasures and marines) is independent of the agent's step (pirates).
            for marines_action in all_marines_possible_locations_combinations:
                for treasures_action in all_treasures_possible_locations_combinations:
                    new_state_reward = 0  # Reward for doing the given action and reaching the next state.
                    prob = 1  # Probability of reaching this state given this action.
                    pirates_info = tuple()
                    treasures_locations = tuple()
                    marines_locations = tuple()
            
                    # For each pirate, we calculate the next location and capacity given the action and the current state.
                    for pirate, pirate_action in zip(state[0], action):
                        capacity = pirate[2]  # If the action is not 'collect', 'deposit' or being caught by a marine, the capacity doesn't change.
                        pirate_next_location = pirate[1]  # If action is not 'sail', pirate_next_location doesn't change.
                        
                        # If the action is 'sail', we update the next location of the pirate.
                        if pirate_action[0] == 'sail':
                            pirate_next_location = pirate_action[2]

                        # If the action is 'deposit', we add the reward for depositing the treasure to the new state reward, and update the capacity of the pirate.
                        if pirate_action[0] == 'deposit':
                            new_state_reward += DROP_IN_DESTINATION_REWARD * (2 - capacity)
                            capacity = 2

                        caught_by_marine = False
                        # For each marine, we check if the pirate is caught by a marine in the next location.
                        for marine_action in marines_action:
                            marine_loc = self.marines_paths[marine_action[0]][marine_action[1]]
                            prob *= marine_action[2]  # We multiply the probability of this action and next state by the probability of the marine's movement.
                            marines_locations += ((marine_action[0], marine_action[1]),)  # We add the new location of the marine to the new state.
                            
                            # If the pirate is not depositing a treasure and the next location of the pirate is the same as the location of a marine, the pirate is caught by the marine.
                            if (pirate_action[0] != 'deposit') and (pirate_next_location == marine_loc):
                                caught_by_marine = True
                        
                        # If the pirate is not depositing a treasure and the action is 'collect', we update the capacity of the pirate.
                        if (not caught_by_marine) and (pirate_action[0] == 'collect'):
                            capacity -= 1

                        # If the pirate is caught by a marine, we add the penalty for being caught by a marine to the new state reward, and update the capacity of the pirate.
                        if caught_by_marine:
                            new_state_reward += MARINE_COLLISION_PENALTY
                            capacity = 2

                        pirates_info += ((pirate[0], pirate_next_location, capacity),)  # We add the new location and capacity of the pirate to the new state.
                    
                    # For each treasure, we calculate the next location of the treasure given the action and the current state.
                    for treasure_action in treasures_action:
                        prob *= treasure_action[2]  # We multiply the probability of this action and next state by the probability of the treasure's movement.
                        treasures_locations += ((treasure_action[0], treasure_action[1]),)

                    # We add the new state to the list of all next states, with the probability of reaching this state given this action, and the reward for doing the given action and reaching the next state.
                    next_states_list.append(((pirates_info, treasures_locations, marines_locations), prob, new_state_reward))
        # We add the list of all next states to a dictionary, so that we don't have to calculate it again for the same state and action.
        self.next_states[(state, action)] = next_states_list


    def updating_value(self, state, t):
        """ Update the value of the given state for the given time step using the value iterations algorithm. """
        max_value_and_action = (float('-inf'), None)  # The maximum value and action for the given state and time step, initialized to the smallest possible value.
        for action in self.actions(state): # Looping on all possible actions of the given state
            # If the action is 'terminate', we set the maximum value and action to 0 and 'terminate' respectively.
            if action == 'terminate':
                max_value_and_action = (self.value_iterations[state][t-1][0], 'terminate') #[(self.value_iterations[state][t-1][0], 'terminate')]
                continue
            # If the next states of the given state and action are not already calculated, we calculate them.
            if (state, action) not in self.next_states:
                self.find_next_states(state, action)

            next_states_list = self.next_states[(state, action)]
            # We calculate the value of the given state and action using the value iterations algorithm,
            # With a slight change, because the reward is on reaching the next state and depends on the action, instead of being on the current state.
            value_of_action = sum(next_state[1] * (next_state[2] + self.value_iterations[next_state[0]][t-1][0]) for next_state in next_states_list)

            # If the value of the action is greater than the maximum value, we update the maximum value and action.
            if value_of_action >= max_value_and_action[0]:
                max_value_and_action = (value_of_action, action)
        self.value_iterations[state][t] = max_value_and_action  # We update the value of the given state and time step with the action that maximizes the value.


    def act(self, state):
        state = self.create_state(state)  # Convert the given state to a tuple format according to our implementation.
        action = self.value_iterations[state][self.time_step][1]  # Get the action that maximizes the value of the given state and the current time step.
        self.time_step -= 1
        return action

class PirateAgent:
    def __init__(self, initial):
        self.map = initial['map']
        self.time_step = initial['turns to go']
        self.len_rows = len(self.map)
        self.len_cols = len(self.map[0])
        # base - the location where all pirates start from
        self.base = list(initial['pirate_ships'].values())[0]['location']
        state_dict = dict()

        self.num_pirates = len(initial['pirate_ships'])

        # list of all possible locations on the map
        possible_locations = list()
        for i in range(self.len_rows):
            for j in range(self.len_cols):
                if self.map[i][j] != 'I':
                    possible_locations.append((i, j))
        # We refer to all pirates as one pirate - all pirates will do the same move in a given action as if there is only one pirate in the game
        self.pirates_names = [pirate for pirate in initial['pirate_ships']]
        state_dict['pirates'] = [(location, capacity) for capacity in range(3) for location in possible_locations]

        self.treasures = dict()
        treasures_info = dict()
        for treasure in initial['treasures'].keys():
            self.treasures[treasure] = dict()
            # list of all possible locations for each treasure.
            possible_locations = initial['treasures'][treasure]['possible_locations']
            self.treasures[treasure]['possible_locations'] = possible_locations
            # the probability of changing location for each treasure for each location it can be in
            self.treasures[treasure]['prob_change_location'] = initial['treasures'][treasure]['prob_change_location']/len(possible_locations)
            treasures_info[treasure] = [(treasure, location) for location in initial['treasures'][treasure]['possible_locations']]
        state_dict['treasures'] = list(product(*treasures_info.values()))  # A cartesian product, in order to get all possible combinations of treasures' locations.
        
        self.marines_info = tuple()
        for marine in initial['marine_ships'].keys():
            marine_locations = initial['marine_ships'][marine]['path']
            # the probability of the marine's location to be in each position in its path is uniformly distributed
            prob = 1/len(marine_locations)
            self.marines_info += ((marine_locations, prob),)
       
        all_states = list(product(*state_dict.values()))  # A cartesian product, in order to get all possible combinations of states.
        self.next_states = dict()

        self.value_iterations = dict() # A dictionary of all possible states, with a list of 100 tuples, each tuple contains the action that maximizes the value and the value of the state.
        # self.value_iterations[state][t] = (value of the state, action that maximizes the value)
        
        self.initial_state = self.create_state(initial)

        for t in range(0, 1+initial['turns to go']):
            for state in all_states:
                if t == 0:
                    self.value_iterations[state] = [(0, None)]*(1+initial['turns to go'])
                else:
                    self.updating_value(state, t)

    def is_valid_location(self, x, y):
        """
        This function checks if the given location is valid for the pirates to sail to.
        :param x: the x coordinate of the location
        :param y: the y coordinate of the location
        """
        return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (self.map[x][y] != 'I')
    
    def is_valid_island(self, x, y):
        """
        This function checks if the given location is an island in the map.
        :param x: the x coordinate of the location
        :param y: the y coordinate of the location
        """
        return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (self.map[x][y] == 'I')
    
    def create_state(self, initial):
        """
        This function converts the given state of the game to our state representation.
        :param initial: the current state of the game (a dictionary)
        """
        pirates_info = (initial['pirate_ships'][self.pirates_names[0]]['location'], initial['pirate_ships'][self.pirates_names[0]]['capacity'])

        treasures_locations = tuple()
        for treasure in initial['treasures'].keys():
            treasures_locations += ((treasure, initial['treasures'][treasure]['location']),)
        
        return (pirates_info, treasures_locations)


    def actions(self, state):
        """
        This function returns all possible actions for the given state.
        :param state: the current state of the game (a tuple)
        """
        pirates_info = state[0]
        all_possible_actions = list()

        all_possible_actions.append('terminate')
        all_possible_actions.append('reset')
        all_possible_actions.append(('wait',))

        # all possiblities for the pirates to move to
        for index_change in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x = pirates_info[0][0] + index_change[0]
            new_y = pirates_info[0][1] + index_change[1]
            if self.is_valid_location(new_x, new_y):
                all_possible_actions.append(('sail', (new_x, new_y)))

            # if the pirates are next to a treasure and have a capacity to collect it, they can collect it
            if self.is_valid_island(new_x, new_y) and pirates_info[1] > 0:
                for treasure in state[1]:
                    if (new_x, new_y) == treasure[1]:
                        all_possible_actions.append(('collect', treasure[0]))
        
        # if the pirates are in the base and have treasures, they can deposit them
        if (pirates_info[0] == self.base) and (pirates_info[1] < 2):
            all_possible_actions.append(('deposit',))

        return all_possible_actions
            
    def find_next_states(self, state, action):
        """
        This function finds all possible next states for the given state and action.
        :param state: the current state of the game (a tuple)
        :param action: the action to be taken (a tuple)
        """
        next_states_list = list()

        if action == 'reset':
            prob = 1  # Probability of reacing this state given this action.
            new_state_reward = RESET_PENALTY
            next_states_list.append((self.initial_state, prob, new_state_reward)) # New state: (pirates_info, self.initial_state[1], self.initial_state[2])
        
        elif action == 'terminate':
            prob = 1
            next_states_list.append((None, prob, new_state_reward))  # New state: None (game is over, no next state).
        
        else:
            treasures_info = {treasure[0]: list() for treasure in state[1]}
            # creating possible states for the treasures
            for treasure in state[1]:
                for possible_location in self.treasures[treasure[0]]['possible_locations']:
                    prob = self.treasures[treasure[0]]['prob_change_location']
                    if possible_location == treasure[1]:
                        # the probability of the treasure of staying in its current location (probability to move to this location or staying there)
                        prob += (1 - prob*len(self.treasures[treasure[0]]['possible_locations']))
                    treasures_info[treasure[0]].append((treasure[0], possible_location, prob))
            all_treasures_possible_locations_combinations = list(product(*treasures_info.values()))  # A cartesian product, in order to get all possible combinations of treasures' locations.
            pirates = state[0]

            for treasures_action in all_treasures_possible_locations_combinations:
                new_state_reward = 0
                prob = 1
                pirates_info = tuple()
                treasures_locations = tuple()
        
                capacity = pirates[1]
                pirate_next_location = pirates[0]  # If action is not 'sail', pirate_next_location doesn't change.
                if action[0] == 'sail':
                    pirate_next_location = action[1]

                if action[0] == 'deposit':
                    # all pirates gets the reward for depositing the treasures
                    new_state_reward += DROP_IN_DESTINATION_REWARD * (2 - capacity) * self.num_pirates
                    capacity = 2

                caught_by_marine = 0
                for marine in self.marines_info:                    
                    if (action[0] != 'deposit') and (pirate_next_location in marine[0]):
                        # the probability of the marine to be in the location of the pirate
                        caught_by_marine += marine[1]
                            
                if (not caught_by_marine) and (action[0] == 'collect'):
                    capacity -= 1

                if caught_by_marine:
                    # sum of the penalties for all pirates if caught by any marine w.r.t the sum of probabilities of the marines they encountered to be in the location of the pirate (the pirate represents all pirates)
                    new_state_reward += MARINE_COLLISION_PENALTY * self.num_pirates * caught_by_marine
                    capacity = 2

                pirates_info = (pirate_next_location, capacity)
                
                for treasure_action in treasures_action:
                    prob *= treasure_action[2]
                    treasures_locations += ((treasure_action[0], treasure_action[1]),)

                # appending the new state to the list of next states
                next_states_list.append(((pirates_info, treasures_locations), prob, new_state_reward))
        # saving the next states for the given state and action in a dictionary (later will save time for the same state and action)
        self.next_states[(state, action)] = next_states_list


    def updating_value(self, state, t):
        """
        This function updates the value of the given state at time t.
        :param state: the current state of the game (a tuple)
        :param t: the current time step in the game
        """
        max_value_and_action = (float('-inf'), None)
        for action in self.actions(state): # all possible actions of the state
            if action == 'terminate':
                # if the action is terminate, the value of the state is 0 and the action is terminate
                max_value_and_action = (self.value_iterations[state][t-1][0], 'terminate')
                continue
            if (state, action) not in self.next_states:
                self.find_next_states(state, action)

            next_states_list = self.next_states[(state, action)]
            value_of_action = sum(next_state[1] * (next_state[2] + self.value_iterations[next_state[0]][t-1][0]) for next_state in next_states_list)

            # updating the maximum value and its action for the state at time t
            if value_of_action >= max_value_and_action[0]:
                max_value_and_action = (value_of_action, action)
        self.value_iterations[state][t] = max_value_and_action  


    def act(self, state):
        """
        This function returns the action to be taken for the given state.
        :param state: the current state of the game (a dictionary)
        """
        state = self.create_state(state)
        # taking the action that maximizes the value of the state at the current time step
        action = self.value_iterations[state][self.time_step][1]
        # assigning the actions as required for the game
        if type(action) == tuple:
            length = len(action)
            action_all_pirates = tuple()
            for pirate in self.pirates_names:
                if length == 1:
                    action_all_pirates += ((action[0], pirate),)
                else:
                    action_all_pirates += ((action[0], pirate, action[1]),)
        else:
            action_all_pirates = action
        # decreasing the time step
        self.time_step -= 1
        return action_all_pirates

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

        # list of all possible locations on the map
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
        """ Find all next states we can get to from the given state and action.

            @param state: A tuple representing the current state.
            @param action: A tuple representing the action we took from the current state.
            
            @return: A list of all possible next states given this state and action,
                     where each next state is represented by a tuple of (state, probability, reward).
        """
        next_states_list = list()
        pirates_info = tuple()

        # If the action is 'reset', we add the initial state to the list of all next states.
        if action == 'reset':
            prob = 1  # Probability of reaching the initial state with the action 'reset'.
            new_state_reward = RESET_PENALTY  # Reward of making the action 'reset'.
            next_states_list.append((self.initial_state, prob, new_state_reward)) # New state: (pirates_info, self.initial_state[1], self.initial_state[2])
        
        # If the action is 'terminate', we add None to the list of all next states (game is over, no next state).
        elif action == 'terminate':
            prob = 1  # Probability of reacing None given this action.
            next_states_list.append((None, prob, new_state_reward))  # New state: None, Reward: 0.
        
        else:
            marines_info = {marine[0]: list() for marine in state[2]}  # A dictionary of all possible indices in the path and the probability of it, for each marine.
            for marine in state[2]:
                # If the marine has only one location in its path, we add it to the list of all possible indices with probability 1.
                if len(self.marines_paths[marine[0]]) == 1:
                    marines_info[marine[0]].append((marine[0], marine[1], 1))
                # If the marine is in the first location in its path, we add it to the list of all possible indices with probability 1/2, and also probability 1/2 to the next location in the path.
                elif marine[1] == 0:
                    marines_info[marine[0]].append((marine[0], marine[1], 1/2))
                    marines_info[marine[0]].append((marine[0], marine[1]+1, 1/2))
                # If the marine is in the last location in its path, we add it to the list of all possible indices with probability 1/2, and also probability 1/2 to the previous location in the path.
                elif marine[1] == len(self.marines_paths[marine[0]]) - 1:
                    marines_info[marine[0]].append((marine[0], marine[1], 1/2))
                    marines_info[marine[0]].append((marine[0], marine[1]-1, 1/2))
                # If the marine is in any other location in its path, we add it to the list of all possible indices with probability 1/3, and also probability 1/3 to the previous and next locations in the path.
                else:
                    marines_info[marine[0]].append((marine[0], marine[1], 1/3))
                    marines_info[marine[0]].append((marine[0], marine[1]-1, 1/3))
                    marines_info[marine[0]].append((marine[0], marine[1]+1, 1/3))
            all_marines_possible_locations_combinations = list(product(*marines_info.values()))  # A cartesian product, in order to get all possible combinations of marines' locations.

            treasures_info = {treasure[0]: list() for treasure in state[1]}
            # For each treasure, we add all possible combinations of its name, location and probability of changing location to the list of all possible combinations of treasures' locations.
            for treasure in state[1]:
                for possible_location in self.treasures[treasure[0]]['possible_locations']:
                    prob = self.treasures[treasure[0]]['prob_change_location']
                    if possible_location == treasure[1]:
                        prob += (1 - prob*len(self.treasures[treasure[0]]['possible_locations']))
                    treasures_info[treasure[0]].append((treasure[0], possible_location, prob))
            all_treasures_possible_locations_combinations = list(product(*treasures_info.values()))  # A cartesian product, in order to get all possible combinations of treasures' locations.
  
            # For each combination of marines' locations and treasures' locations, we add all possible combinations of pirates' locations and capacities to the list of all next states.
            # We do that because the environment step (treasures and marines) is independent of the agent's step (pirates).
            for marines_action in all_marines_possible_locations_combinations:
                for treasures_action in all_treasures_possible_locations_combinations:
                    new_state_reward = 0  # Reward for doing the given action and reaching the next state.
                    prob = 1  # Probability of reaching this state given this action.
                    pirates_info = tuple()
                    treasures_locations = tuple()
                    marines_locations = tuple()
            
                    # For each pirate, we calculate the next location and capacity given the action and the current state.
                    for pirate, pirate_action in zip(state[0], action):
                        capacity = pirate[2]  # If the action is not 'collect', 'deposit' or being caught by a marine, the capacity doesn't change.
                        pirate_next_location = pirate[1]  # If action is not 'sail', pirate_next_location doesn't change.
                        
                        # If the action is 'sail', we update the next location of the pirate.
                        if pirate_action[0] == 'sail':
                            pirate_next_location = pirate_action[2]

                        # If the action is 'deposit', we add the reward for depositing the treasure to the new state reward, and update the capacity of the pirate.
                        if pirate_action[0] == 'deposit':
                            new_state_reward += DROP_IN_DESTINATION_REWARD * (2 - capacity)
                            capacity = 2

                        caught_by_marine = False
                        # For each marine, we check if the pirate is caught by a marine in the next location.
                        for marine_action in marines_action:
                            marine_loc = self.marines_paths[marine_action[0]][marine_action[1]]
                            prob *= marine_action[2]  # We multiply the probability of this action and next state by the probability of the marine's movement.
                            marines_locations += ((marine_action[0], marine_action[1]),)  # We add the new location of the marine to the new state.
                            
                            # If the pirate is not depositing a treasure and the next location of the pirate is the same as the location of a marine, the pirate is caught by the marine.
                            if (pirate_action[0] != 'deposit') and (pirate_next_location == marine_loc):
                                caught_by_marine = True

                        # If the pirate is not depositing a treasure and the action is 'collect', we update the capacity of the pirate.   
                        if (not caught_by_marine) and (pirate_action[0] == 'collect'):
                            capacity -= 1

                        # If the pirate is caught by a marine, we add the penalty for being caught by a marine to the new state reward, and update the capacity of the pirate.
                        if caught_by_marine:
                            new_state_reward += MARINE_COLLISION_PENALTY
                            capacity = 2

                        pirates_info += ((pirate[0], pirate_next_location, capacity),)  # We add the new location and capacity of the pirate to the new state.

                    # For each treasure, we calculate the next location of the treasure given the action and the current state.  
                    for treasure_action in treasures_action:
                        prob *= treasure_action[2]  # We multiply the probability of this action and next state by the probability of the treasure's movement.
                        treasures_locations += ((treasure_action[0], treasure_action[1]),)

                    # We add the new state to the list of all next states, with the probability of reaching this state given this action, and the reward for doing the given action and reaching the next state.
                    next_states_list.append(((pirates_info, treasures_locations, marines_locations), prob, new_state_reward))
        self.next_states[(state, action)] = next_states_list  # A dictionary of all next states given this state and action, so that we don't have to calculate it again for the same state and action.


    def updating_value(self, state):
        """ Update the value of the given state for the given time step using the value iterations algorithm. """
        max_value_and_action = (float('-inf'), None)  # The maximum value and action for the given state and time step, initialized to the smallest possible value.
        for action in self.actions(state): # Looping on all possible actions of the given state
            # If the action is 'terminate', we set the maximum value and the corresponding action to the value we got up until now and 'terminate' respectively.
            if action == 'terminate':
                max_value_and_action = (0, 'terminate')
                continue
            if (state, action) not in self.next_states:  # If the next states of the given state and action are not already calculated, we calculate them.
                self.find_next_states(state, action)

            next_states_list = self.next_states[(state, action)]  # all possible next states given this state and action
            # We calculate the value of the given state and action using the value iterations algorithm,
            # With a slight change, because the reward is on reaching the next state and depends on the action, instead of being on the current state.
            value_of_action = self.gamma * sum(next_state[1] * (next_state[2] + self.value_iterations[next_state[0]][0]) for next_state in next_states_list)

            # If the value of the action is greater than the maximum value, we update the maximum value and action.
            if value_of_action >= max_value_and_action[0]:
                max_value_and_action = (value_of_action, action)
        self.value_iterations[state] = max_value_and_action  # We update the value of the given state and time step with the action that maximizes the value.

    def act(self, state):
        state = self.create_state(state)  # Convert the given state to a tuple format according to our implementation.
        action = self.value_iterations[state][1]  # Get the action that maximizes the value of the given state and the current time step.
        print(state)
        print(action)
        print()
        return action

    def value(self, state):
        return self.value_iterations[state][0]  # The value of the action that maximizes the value of the given state.

