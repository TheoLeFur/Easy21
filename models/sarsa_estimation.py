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
            action = Action.to_action(np.argmax(self.Q[state.dealer - 1, state.player - 1, :]))

        return action

    def train(self):
        for episode in tqdm(range(self.num_train_per_epoch)):
            # random start
            s = self.environment.reset()
            a = self.select_action(s)

            while not s.terminal:
                # update N(s,a)
                self.N[s.dealer - 1, s.player - 1, Action.to_int(a)] += 1
                # execute action a and observe s_new, r
                s_new, r = self.environment.step(s, a)
                dealer_id = s.dealer - 1
                player_id = s.player - 1
                if s_new.terminal:
                    Q_new = 0
                else:
                    a_new = self.select_action(s_new)
                    dealer_id_new = s_new.dealer - 1
                    player_id_new = s_new.player - 1
                    Q_new = self.Q[dealer_id_new, player_id_new, Action.to_int(a_new)]
                # using a varying step size alpha = 1/N(st,at)
                alpha = 1.0 / self.N[dealer_id, player_id, Action.to_int(a)]
                # calculate TD error
                td_error = r + self.gamma * Q_new - self.Q[dealer_id, player_id, Action.to_int(a)]
                # update the eligibility trace
                self.E[dealer_id, player_id, Action.to_int(a)] += 1
                # update the Q and E for all state-action pairs
                self.Q += alpha * td_error * self.E
                self.E *= self.gamma * self.lambda_param
                s = s_new
                if not s_new.terminal:
                    a = a_new

        self.episodes += self.num_episodes

    def plot(self, ax):

        x = np.arange(0, self.environment.dealer_value_count, 1)
        y = np.arange(0, self.environment.player_value_count, 1)
        x, y = np.meshgrid(x, y)
        z = self.V[x, y]
        return ax.plot_surface(x, y, z, cmap=cm.bwr, antialiased=False)
