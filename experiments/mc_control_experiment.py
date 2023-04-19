import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

from environments.easy21_env import Environment
from models.mc_estimation import MonteCarloControlAgent


def RunMonteCarloExperiment(params):

    mc_agent = MonteCarloControlAgent(params)
    mc_agent.train()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    mc_agent.plot(ax)
    plt.title("Value function")
    ax.set_xlabel("dealer showing")
    ax.set_ylabel("player showing")
    ax.set_zlabel("Value function")

    ax.set_xticks(range(1, mc_agent.environment.dealer_value_count, 1))
    ax.set_yticks(range(1, mc_agent.environment.player_value_count, 1))
    plt.show()
    plt.savefig(params["fig_dir"])


if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--N0", type = int, default = 100)
    parser.add_argument("--num_episodes", type = int, default = 1e6)


    training_args = dict(
        environment = Environment(),
        episode_log_freq = 1e4,
    )

    args = parser.parse_args()
    kwargs = vars(args)
    training_params = {**training_args, **kwargs}

    RunMonteCarloExperiment(training_params)

    


