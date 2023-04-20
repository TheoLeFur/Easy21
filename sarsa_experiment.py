import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.sarsa_estimation import SarsaAgent
from models.mc_estimation import MonteCarloControlAgent
from environments.easy21_env import Environment
from collections import OrderedDict
from tqdm import tqdm


class Sarsa_Trainer(object):

    def __init__(self, params: dict, true_Q = None) -> None:
        self.params = params
        self.num_epochs = params["num_epochs"]
        self.num_train_per_epoch = params["num_train_per_epoch"]
        self.true_Q = true_Q
        self.agent = SarsaAgent(params)
        self.mse_log = []

    def train(self):
        for epoch in tqdm(range(self.num_epochs)):
            self.agent.train()
            print(f"Epoch number {epoch}")
            mse = np.mean((self.agent.Q - self.true_Q) ** 2)
            self.mse_log.append(mse)


def run_experiment():
    params = dict(
        N0=100,
        num_episodes=int(1e4),
        num_train_per_epoch=1,
        episode_log_freq=10000,
        num_epochs=1000,
        lambda_start=0,
        lambda_end=1,
        number_lambdas=10,
        gamma=1,
        environment=Environment()
    )

    lambda_mse = OrderedDict()
    lambdas = np.linspace(params["lambda_start"], params["lambda_end"], params["number_lambdas"])
    mc_agent = MonteCarloControlAgent(params)

    print("-----------  Training the Monte Carlo Agent  -----------")

    mc_agent.train()
    Q_true = mc_agent.Q
    print(Q_true)

    for lambda_param, lambda_id in enumerate(lambdas):
        print(f"-----------Training Sarsa Agent for param lambda = {lambda_param}  -----------")
        training_args = dict(
            **params,
            **dict(lambda_param=lambda_param)
        )
        trainer = Sarsa_Trainer(training_args, Q_true)
        trainer.train()
        lambda_mse[f"{lambda_id}"] = trainer.mse_log

    plt.figure()
    plt.subplot(211)
    plt.plot(lambdas, [x[-1] for x in lambda_mse.items()])
    plt.show()



if __name__ == "__main__":
    run_experiment()
