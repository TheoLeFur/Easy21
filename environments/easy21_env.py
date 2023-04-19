import random
import copy
from enum import Enum


class Action(Enum):
    """
    Our action space is composed of two actions. 
    We either stick or hit. 
    """
    hit = 1
    stick = 0

    @staticmethod
    def to_action(n: int):
        return Action.hit if n == 1 else Action.stick

    @staticmethod
    def to_int(action):
        return 1 if action == Action.hit else 0


class Card(object):

    def __init__(self, force_black: bool = False):
        self.force_black = force_black
        self.value = random.randint(1, 10)

        if self.force_black or random.randint(1, 3) != 3:
            self.is_black = True
        else:
            self.is_black = False
            self.value = -self.value


class State(object):

    def __init__(self, player, dealer, terminal):
        self.player = player
        self.dealer = dealer
        self.terminal = terminal


class Environment(object):

    def __init__(self):

        self.dealer_value_count = 10
        self.player_value_count = 21
        self.action_count = 2

    def reset(self):

        s = State(Card(True).value, Card(True).value, False)
        return s

    def step(self, state: State, action: Action):

        new_state = copy.copy(state)
        reward = 0

        if action == action.hit:

            new_state.player += Card().value
            if (new_state.player < 1 or new_state.player > 21):
                new_state.terminal = True
                reward = -1

            return new_state, reward

        elif action == Action.stick:

            while not new_state.terminal:

                new_state.dealer += Card().value

                if (new_state.dealer < 1 or new_state.dealer > 21):
                    new_state.terminal = True
                    reward = 1
                elif new_state.dealer > 17:
                    new_state.terminal = True
                    if new_state.player > new_state.dealer:
                        reward = 1
                    elif new_state.player < new_state.dealer:
                        reward = -1

        return new_state, reward
