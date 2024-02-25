ids = ["322620873", "314779166"]

RESET_PENALTY = 2
DROP_IN_DESTINATION_REWARD = 4
MARINE_COLLISION_PENALTY = 1

from itertools import product

class OptimalPirateAgent:
    def __init__(self, initial):
        self.initial = initial

    def act(self, state):
        raise NotImplemented


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
