ids = ["111111111", "222222222"]


class OptimalPirateAgent:
    def __init__(self, initial):
        self.initial = initial

    def act(self, state):
        raise NotImplemented


class PirateAgent:
    def __init__(self, initial):
        self.initial = initial

    def act(self, state):
        raise NotImplemented


class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma

    def act(self, state):
        raise NotImplemented

    def value(self, state):
        raise NotImplemented
