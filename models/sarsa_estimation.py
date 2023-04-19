import numpy as np
import random
import matplotlib.cm as cm
from models.abstract_model import BaseModel
from environments.easy21_env import Action
from environments.easy21_env import State
from environments.easy21_env import Environment
from tqdm import tqdm



class SarsaAgent(BaseModel):

    def __init__(self, params: dict) -> None:

        self.N0 = params["N0"]
        self.environment = params["environment"]
        self.num_episodes = params["num_episodes"]
        self.lambda_param = params["lambda_param"]
        self.gamma = params["gamma"]
        self.num_train_per_epoch = params["num_train_per_epoch"]

        self.V = np.zeros(shape=(
            self.environment.dealer_value_count, self.environment.player_value_count))
        self.Q = np.zeros(shape=(
            self.environment.dealer_value_count, self.environment.player_value_count, self.environment.action_count))
        self.N = np.zeros(shape=(
            self.environment.dealer_value_count, self.environment.player_value_count, self.environment.action_count))
        self.E = np.zeros(shape=(
            self.environment.dealer_value_count, self.environment.player_value_count, self.environment.action_count))

        self.returns = []
        for _ in range(self.environment.dealer_value_count * self.environment.player_value_count):
            G = []
            for _ in range(self.environment.action_count):
                G.append([])
            self.returns.append(G)

        self.count_wins = 0
        self.episodes = 0

    def select_action(self, state: State) -> None:

        epsilon = self.N0 / \
                  (self.N0 +
                   np.sum(self.N[state.dealer - 1, state.player - 1, :]))
        if random.random() < epsilon:
            if random.random() < 0.5:
                action = Action.hit
            else:
                action = Action.stick
        else:
            action = Action.to_action(
                np.argmax(self.Q[state.dealer - 1, state.player - 1, :]))
        return action

    def train(self):

        for _ in tqdm(range(int(self.num_train_per_epoch))):

            state = self.environment.reset()
            action = self.select_action(state)
            score = 0

            while not state.terminal:

                dealer_idx = state.dealer - 1
                player_idx = state.player - 1
                action_idx = Action.to_int(action)

                self.N[dealer_idx, player_idx, action_idx] += 1
                next_state, reward = self.environment.step(state, action)

                if next_state.terminal:
                    Q_new = 0


                else:
                    new_action = self.select_action(next_state)
                    Q_new = self.Q[next_state.dealer - 1, next_state.player -
                                   1, Action.to_int(new_action)]

                alpha = 1. / self.N[state.dealer - 1, state.player - 1, Action.to_int(action)]
                delta = reward + self.gamma * \
                        Q_new - self.Q[state.dealer - 1,
                                       state.player - 1, Action.to_int(action)]

                self.E[state.dealer - 1, state.player -
                       1, Action.to_int(action)] += 1
                self.Q += alpha * delta * self.E
                self.E *= self.gamma * self.lambda_param

                state = next_state
                score += reward
                if not next_state.terminal:
                    action = new_action

                self.count_wins = self.count_wins + 1 if reward == 1 else self.count_wins

        self.episodes += self.num_episodes

    def plot(self, ax):

        x = np.arange(0, self.environment.dealer_value_count, 1)
        y = np.arange(0, self.environment.player_value_count, 1)
        x, y = np.meshgrid(x, y)
        z = self.V[x, y]
        return ax.plot_surface(x, y, z, cmap=cm.bwr, antialiased=False)
