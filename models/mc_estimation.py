import random 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from environments.easy21_env import *
from models.abstract_model import BaseModel 
from tqdm import tqdm 



class MonteCarloControlAgent(BaseModel):


    def __init__(self, params):
     
        self.N0 = params["N0"]
        self.environment = params["environment"]
        self.num_episodes = params["num_episodes"]
        self.episode_log_freq = params["episode_log_freq"]


        self.V = np.zeros(shape = (self.environment.dealer_value_count, self.environment.player_value_count))
        self.Q = np.zeros(shape=(
            self.environment.dealer_value_count, self.environment.player_value_count, self.environment.action_count))
        self.N = np.zeros(shape=(
            self.environment.dealer_value_count, self.environment.player_value_count, self.environment.action_count))
        
        self.returns = []
        for _ in range(self.environment.dealer_value_count * self.environment.player_value_count):
            G = []
            for _ in range(self.environment.action_count):
                G.append([])
            self.returns.append(G)


        self.count_wins = 0
        self.episodes = 0
    

    def select_action(self, state):
        
        epsilon = self.N0/(self.N0 + np.sum(self.N[state.dealer - 1, state.player - 1, :]))
        if random.random() < epsilon:
            if random.random() < 0.5:
                action = Action.hit
            else:
                action = Action.stick 
        else:
            action = Action.to_action(np.argmax(self.Q[state.dealer-1, state.player-1, :]))
        return action 
    


    def train(self):

        for episode in tqdm(range(int(self.num_episodes))):

            state = self.environment.reset()
            episode_pairs = []
            score = 0

            while not state.terminal:

                action = self.select_action(state)
                self.N[state.dealer - 1, state.player-1, Action.to_int(action)]+=1
                episode_pairs.append((state, action))
                state, reward = self.environment.step(state, action)
                score += reward
            
            self.count_wins += 1 if reward == 1 else self.count_wins

            if episode % self.episode_log_freq == 0:
                print(f"Episode number : {episode}, Episode Score : {score}")
            
            for state, action in episode_pairs:

                idx = self.environment.dealer_value_count * (state.dealer - 1) + state.player
                self.returns[idx][Action.to_int(action)].append(reward)
                error = np.mean(self.returns[idx][Action.to_int(action)] - self.Q[state.dealer - 1, state.player - 1, Action.to_int(action)])
                alpha = 1./self.N[state.dealer - 1, state.player - 1, Action.to_int(action)]
                self.Q[state.dealer - 1, state.player - 1, Action.to_int(action)] += alpha * error 
                self.V[state.dealer - 1, state.player - 1] = np.max(self.Q[state.dealer - 1, state.player - 1, :])


        self.episodes += self.num_episodes

    

    def plot(self, ax):

        x = np.arange(0, self.environment.dealer_value_count, 1)
        y = np.arange(0, self.environment.player_value_count, 1)
        x, y = np.meshgrid(x, y)
        z = self.V[x, y]
        return ax.plot_surface(x, y, z, cmap=cm.bwr, antialiased=False)
    










            




