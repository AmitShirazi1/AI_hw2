ids = ["322620873", "314779166"]


class OptimalPirateAgent:
    def __init__(self, initial):
        self.initial = initial

    def act(self, state):
        raise NotImplemented


class PirateAgent:
    def __init__(self, initial):
        self.initial = initial

    def actions():
        return tuple_of_actions

    def act(self, state):
        turns_to_go = state['turns_to_go'] if state['infinite'] is False else None
        for action in 


class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma

    def act(self, state):
        raise NotImplemented

    def value(self, state):
        raise NotImplemented
