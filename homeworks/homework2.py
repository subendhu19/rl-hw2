from rl687.environments.gridworld import Gridworld
from rl687.environments.cartpole import Cartpole
from rl687.agents.cem import CEM
from rl687.agents.fchc import FCHC
from rl687.agents.ga import GA
from rl687.policies.tabular_softmax import TabularSoftmax
from rl687.policies.linear_approximation import LinearApproximation
import numpy as np
from scipy.special import softmax
from multiprocessing import Pool
import matplotlib.pyplot as plt


def run_gridworld_episode(p):
    environment = Gridworld()
    policy = TabularSoftmax(25, 4)
    policy.parameters = p
    is_end = False
    discounted_return = 0
    t = 0
    while not is_end:
        action = policy.samplAction(environment.state)
        new_state, reward, is_end = environment.step(action)
        discounted_return += (environment.gamma ** t) * reward
        t += 1
    environment.reset()
    return discounted_return


def run_cartpole_episode(p):
    environment = Cartpole()
    policy = LinearApproximation(4, 2)
    policy.parameters = p
    is_end = False
    discounted_return = 0
    t = 0
    while not is_end:
        action = policy.samplAction(environment.state)
        new_state, reward, is_end = environment.step(action)
        discounted_return += (environment.gamma ** t) * reward
        t += 1
    environment.reset()
    return discounted_return


def problem1():
    """
    Apply the CEM algorithm to the More-Watery 687-Gridworld. Use a tabular 
    softmax policy. Search the space of hyperparameters for hyperparameters 
    that work well. Report how you searched the hyperparameters, 
    what hyperparameters you found worked best, and present a learning curve
    plot using these hyperparameters, as described in class. This plot may be 
    over any number of episodes, but should show convergence to a nearly 
    optimal policy. The plot should average over at least 500 trials and 
    should include standard error or standard deviation error bars. Say which 
    error bar variant you used. 
    """
    all_returns = []

    def evaluate(p, episodes):
        pool = Pool(3)
        returns = pool.map(run_gridworld_episode, [p] * episodes)
        for r in returns:
            all_returns.append(r)

        return np.mean(returns)

    agent_policy = TabularSoftmax(25, 4)
    agent = CEM(agent_policy.parameters, 5, 10, 5, 10, evaluate)
    agent_policy.parameters = agent.train()
    print(softmax(agent_policy.parameters.reshape(25, 4), axis=1))


def problem2():
    """
    Repeat the previous question, but using first-choice hill-climbing on the 
    More-Watery 687-Gridworld domain. Report the same quantities.
    """
    all_returns = []

    plt.axis([0, 4000, -20, 4])

    def evaluate(p, episodes):
        pool = Pool(4)
        returns = pool.map(run_gridworld_episode, [p] * episodes)
        for r in returns:
            all_returns.append(r)
            plt.scatter(len(all_returns), r, c='blue')
            plt.pause(0.05)

        return np.mean(returns)

    agent_policy = TabularSoftmax(25, 4)
    agent = FCHC(agent_policy.parameters, 2.5, evaluate)
    plt.show()
    agent_policy.parameters = agent.train()
    print(softmax(agent_policy.parameters.reshape(25, 4), axis=1))


def problem3():
    """
    Repeat the previous question, but using the GA (as described earlier in 
    this assignment) on the More-Watery 687-Gridworld domain. Report the same 
    quantities.
    """
    all_returns = []

    def evaluate(p, episodes):
        pool = Pool(4)
        returns = pool.map(run_gridworld_episode, [p] * episodes)
        for r in returns:
            all_returns.append(r)

        return np.mean(returns)

    agent_policy = TabularSoftmax(25, 4)
    agent = GA(10, evaluate, num_params=100)
    agent_policy.parameters = agent.train()
    print(softmax(agent_policy.parameters.reshape(25, 4), axis=1))


def problem4():
    """
    Repeat the previous question, but using the cross-entropy method on the 
    cart-pole domain. Notice that the state is not discrete, and so you cannot 
    directly apply a tabular softmax policy. It is up to you to create a 
    representation for the policy for this problem. Consider using the softmax 
    action selection using linear function approximation as described in the notes. 
    Report the same quantities, as well as how you parameterized the policy. 
    
    """
    all_returns = []

    def evaluate(p, episodes):
        pool = Pool(4)
        returns = pool.map(run_cartpole_episode, [p] * episodes)
        for r in returns:
            all_returns.append(r)

        return np.mean(returns)

    agent_policy = LinearApproximation(4, 2)
    agent = CEM(agent_policy.parameters, 5, 20, 10, 25, evaluate)
    agent_policy.parameters = agent.train()
    print(agent_policy.parameters.reshape(2, 256))


def problem5():
    """
    Repeat the previous question, but using first-choice hill-climbing (as 
    described in class) on the cart-pole domain. Report the same quantities 
    and how the policy was parameterized. 
    
    """
    all_returns = []

    def evaluate(p, episodes):
        pool = Pool(4)
        returns = pool.map(run_cartpole_episode, [p] * episodes)
        for r in returns:
            all_returns.append(r)

        return np.mean(returns)

    agent_policy = LinearApproximation(4, 2)
    agent = FCHC(agent_policy.parameters, 2.5, evaluate)
    agent_policy.parameters = agent.train()
    print(agent_policy.parameters.reshape(2, 256))


def problem6():
    """
    Repeat the previous question, but using the GA (as described earlier in 
    this homework) on the cart-pole domain. Report the same quantities and how
    the policy was parameterized. 
    """
    all_returns = []

    def evaluate(p, episodes):
        pool = Pool(4)
        returns = pool.map(run_cartpole_episode, [p] * episodes)
        for r in returns:
            all_returns.append(r)

        return np.mean(returns)

    agent_policy = LinearApproximation(4, 2)
    agent = GA(20, evaluate, num_params=512)
    agent_policy.parameters = agent.train()
    # print(agent_policy.parameters.reshape(2, 256))


def main():
    problem1()
    # problem2()
    # problem3()
    # problem4()
    # problem5()
    # problem6()


if __name__ == "__main__":
    main()
