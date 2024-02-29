ids = ["322620873", "314779166"]

RESET_PENALTY = -2
DROP_IN_DESTINATION_REWARD = 4
MARINE_COLLISION_PENALTY = -1

from itertools import product

class State:
    def __init__(self, initial):
        self.pirate_ships = initial["pirate_ships"]
        self.treasures = initial["treasures"]['location']
        self.marine_ships = initial["marine_ships"]
        self.next_states 

class OptimalPirateAgent:
    def __init__(self, initial):
        self.map = initial['map']

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
        for marine in initial['marine_ships'].keys():
            marines_locations += ((marine, initial['marine_ships'][marine]['index'], initial['marine_ships'][marine]['path']),)
        
        self.all_states = dict()
        self.all_states[(pirates_info, treasures_locations, marines_locations)] = tuple()  # ((action1, value_of_action1), (action2, value_of_action2), ...)

        self.len_rows = self.map.shape[0]
        self.len_cols = self.map.shape[1]
        def is_valid_location(self, x, y):
            return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (self.map[x][y] != 'I')

        def actions(self, state):
            pirates_actions = dict()
            pirates_info = state[0]
            for pirate in pirates_info:
                pirates_actions[pirate] = list()
                for index_change in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_x = pirate[1][0] + index_change[0]
                    new_y = pirate[1][1] + index_change[1]
                    if self.is_valid_location(new_x, new_y):
                        pirates_actions[pirate].append(('sail', pirate[0], (new_x, new_y)))

                    if pirate[2] > 0 and self.map[new_x][new_y] == 'I':
                        for treasure in state[1]:
                            if (new_x, new_y) == treasure[1]:
                                pirates_actions[pirate].append(('collect', pirate[0], treasure))
                if (pirate[1] == self.base) and (pirate[2] < 2):
                    pirates_actions[pirate].append(('deposit', pirate[0]))
                pirates_actions[pirate].append(('wait', pirate[0]))
                
            all_possible_actions = list(product(pirate_actions for pirate_actions in pirates_actions.values()))
            all_possible_actions.append('reset')
            all_possible_actions.append('terminate')
            return all_possible_actions

        state # TODO: Recieve state from somewhere.

        def sum_values_of_next_states(self, state, action):
            new_state_value = 0
            if action == 'reset':
                new_state_value = RESET_PENALTY
            # If action == 'terminate', new_state_value doesn't change.
            else:
                for pirate in state[0]:
                    pirate_next_location = pirate[1]
                    if action[0] == 'deposit':
                        new_state_value += DROP_IN_DESTINATION_REWARD * (2 - pirate[2])
                    else:
                        if action[0] == 'sail':
                            pirate_next_location = action[2]

                        if pirate_next_location in state[2][2]:  # If pirate_next_location is in the marine's path.
                            p = 0
                            for marine in state[2]:
                                if len(marine[2]) == 1:
                                    if pirate_next_location == marine[2][0]:
                                        p = 1
                                        break
                                if marine[1] == 0:
                                    if (pirate_next_location == marine[2][0]) or (pirate_next_location == marine[2][1]):
                                        p += 1/2
                                elif marine[1] == len(marine[2])-1:
                                    if (pirate_next_location == marine[2][-1]) or (pirate_next_location == marine[2][-2]):
                                        p += 1/2
                                else:
                                    if (pirate_next_location == marine[2][marine[1]]) or (pirate_next_location == marine[2][marine[1]-1]) or (pirate_next_location == marine[2][marine[1]+1]):
                                        p += 1/3
                            new_state_value += p * MARINE_COLLISION_PENALTY
            
            # TODO: Move everything in state, like treasures and marines, to the next state, and update the self.all_states dictionary.
            return new_state_value
        
""" GPT Code: """
import numpy as np
class PirateGame:
    def _init_(self, num_pirates, num_treasures, num_marines):
        self.num_pirates = num_pirates
        self.num_treasures = num_treasures
        self.num_marines = num_marines
        self.state = self.initialize_game()

    def initialize_game(self):
        # Placeholder function to initialize the game state
        # Implement this based on your actual game representation
        pass

    def get_possible_actions(self):
        # Placeholder function to return possible actions for a pirate
        # Implement this based on your actual game representation
        pass

    def take_action(self, pirate, action):
        # Placeholder function to update the game state based on the chosen action
        # Implement this based on your actual game representation
        pass

    def calculate_reward(self, pirate, action):
        # Placeholder function to calculate the reward for a given action
        # Implement this based on your reward structure
        pass

    def q_learning(game, num_episodes=1000, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        # Q-table initialization
        q_table = np.zeros((game.num_pirates, game.num_treasures, game.num_marines, len(game.get_possible_actions())))

        for episode in range(num_episodes):
            state = game.initialize_game()  # Reset the game state for each episode

            for pirate in range(game.num_pirates):
                while not game.is_game_over():
                    # Choose action epsilon-greedy
                    if np.random.rand() < exploration_prob:
                        action = np.random.choice(game.get_possible_actions())
                    else:
                        action = np.argmax(q_table[pirate, state[pirate]])

                    # Take the chosen action
                    next_state = game.take_action(pirate, action)

                    # Calculate reward
                    reward = game.calculate_reward(pirate, action)

                    # Update Q-value
                    best_next_action = np.argmax(q_table[pirate, next_state[pirate]])
                    q_table[pirate, state[pirate], action] += learning_rate * (
                        reward + discount_factor * q_table[pirate, next_state[pirate], best_next_action]
                        - q_table[pirate, state[pirate], action]
                    )

                    # Move to the next state
                    state = next_state

            return q_table

        # Example Usage
        num_pirates = 2
        num_treasures = 3
        num_marines = 2

        game = PirateGame(num_pirates, num_treasures, num_marines)
        q_table = q_learning(game)

        # Use the learned Q-table for decision-making during gameplay
        # (This part heavily depends on your actual game implementation)
        # You might need additional functions to convert state to Q-table indices and vice versa
""" End of GPT code. """


    def act(self, state):
        raise NotImplemented

# TODO: move to optimal, change act to O(1) and init can be very large.
#       Calculate all possible routes.
#       in non-optimal we have a complicated board, can't do the same.
class PirateAgent:
    def __init__(self, initial):
        self.initial = initial
        self.len_rows = initial["map"].shape[0]
        self.len_cols = initial["map"].shape[1]
        base_row = [x for x in self.initial['map'] if 'B' in x][0]
        self.base = (self.initial['map'].index(base_row), base_row.index('B'))
        self.marine_tracks = []
        for marine in self.initial['marine_ships'].keys():
            self.marine_tracks += marine['path']
        self.num_marines = len(self.initial['marine_ships'])

    def is_valid_location(self, x, y):
        return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (self.initial['map'][x][y] != 'I')
        

    def actions(self, state):
        pirates_actions = {pirate: [] for pirate in state["pirate_ships"]}
        for pirate in state["pirate_ships"]:
            for index_change in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = state["pirate_ships"][pirate]["location"][0] + index_change[0]
                new_y = state["pirate_ships"][pirate]["location"][1] + index_change[1]
                if self.is_valid_location(new_x, new_y):
                    pirates_actions[pirate].append(('sail', pirate, (new_x, new_y)))

                if state["pirate_ships"][pirate]["capacity"] > 0 and self.initial["map"][new_x][new_y] == 'I':
                    for treasure in state["treasures"]:
                        if (new_x, new_y) == treasure['location']:
                            pirates_actions[pirate].append(('collect', pirate, treasure))
            if (pirate['location'] == self.base) and (pirate['capacity'] < 2):
                pirates_actions[pirate].append(('deposit', pirate))
            pirates_actions[pirate].append(('wait', pirate))
            
        all_possible_actions = list(product(pirate_actions for pirate_actions in pirates_actions.values()))
        all_possible_actions.append('reset')
        all_possible_actions.append('terminate')
        return all_possible_actions

    def act(self, state):
        max_action = None
        max_action_value = float('-inf')

        for action in self.actions(state):
            accumulated_value = 0
            if action == 'reset':
                accumulated_value -= RESET_PENALTY
                # TODO: think about the good things in reset
            else:
                for pirate_action in action:
                    pirate_next_location = state["pirate_ships"][pirate_action[1]]["location"]
                    if pirate_action[0] == 'deposit':  # As for now, in danger of being caught by marine, and also when collecting treasure.
                        accumulated_value += DROP_IN_DESTINATION_REWARD * (2 - state["pirate_ships"][pirate_action[1]]["capacity"])
                    if pirate_action[0] == 'sail':
                        pirate_next_location = pirate_action[2]
                    if pirate_next_location in self.marine_tracks:
                        p = 0
                        for marine in self.initial['marine_ships'].values():
                            if len(marine["path"]) == 1:
                                if pirate_next_location == marine["path"][0]:
                                    p = self.num_marines
                                    break
                            if marine['index'] == 0:
                                if (pirate_next_location == marine["path"][0]) or (pirate_next_location == marine["path"][1]):
                                    p += 1/2
                            elif marine['index'] == len(marine["path"])-1:
                                if (pirate_next_location == marine["path"][-1]) or (pirate_next_location == marine["path"][-2]):
                                    p += 1/2
                            else:
                                if (pirate_next_location == marine["path"][marine['index']]) or (pirate_next_location == marine["path"][marine['index']-1]) or (pirate_next_location == marine["path"][marine['index']+1]):
                                    p += 1/3                                
                        accumulated_value -= MARINE_COLLISION_PENALTY * (p/self.num_marines)
                        # TODO: take another look at the probabilities.
                if accumulated_value > max_action_value:
                    max_action_value = accumulated_value
                    max_action = action
        
        if max_action_value < 0:
            max_action == 'terminate'
            # TODO: think about when to use terminate: when actual value is negative, or when the best action is negative. 
        
        # return reward + max(actions(state))


class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma

    def act(self, state):
        raise NotImplemented

    def value(self, state):
        raise NotImplemented
