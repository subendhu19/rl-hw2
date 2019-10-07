from rl687.environments.gridworld import Gridworld
from rl687.environments.cartpole import Cartpole
from rl687.agents.cem import CEM
from rl687.agents.fchc import FCHC
from rl687.agents.ga import GA
from rl687.policies.tabular_softmax import TabularSoftmax
from rl687.policies.linear_approximation import LinearApproximation
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


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
        if t > 200:
            discounted_return = -50
            break
    environment.reset()
    return discounted_return


def run_cartpole_episode(p, basis):
    environment = Cartpole()
    policy = LinearApproximation(state_dim=4, num_actions=2, basis=basis)
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


def init_gridworld_population(size: int) -> np.ndarray:
    return np.random.normal(size=(size, 100))


def init_cartpole_population_2(size: int) -> np.ndarray:
    return np.random.normal(size=(size, 162))


def init_cartpole_population_3(size: int) -> np.ndarray:
    return np.random.normal(size=(size, 512))


def problem1(config, iterations: int=200):
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
        returns = []
        for i in range(episodes):
            r = run_gridworld_episode(p)
            returns.append(r)
            all_returns.append(r)

        return np.mean(returns)

    agent_policy = TabularSoftmax(25, 4)
    agent = CEM(agent_policy.parameters, sigma=config[0], popSize=config[1], numElite=config[2], numEpisodes=config[3],
                evaluationFunction=evaluate, epsilon=config[4])
    bar = range(iterations)
    for i in bar:
        agent_policy.parameters = agent.train()
        # bar.set_description("Average return: {}".format(evaluate(agent_policy.parameters, 5)))
    return np.array(all_returns)


def problem2(config, iterations: int=1000):
    """
    Repeat the previous question, but using first-choice hill-climbing on the 
    More-Watery 687-Gridworld domain. Report the same quantities.
    """
    all_returns = []

    def evaluate(p, episodes):
        returns = []
        for i in range(episodes):
            r = run_gridworld_episode(p)
            returns.append(r)
            all_returns.append(r)

        return np.mean(returns)

    agent_policy = TabularSoftmax(25, 4)
    agent = FCHC(agent_policy.parameters, sigma=config[0], evaluationFunction=evaluate, numEpisodes=config[1])
    bar = range(iterations)
    for i in bar:
        agent_policy.parameters = agent.train()
        # bar.set_description("Average return: {}".format(evaluate(agent_policy.parameters, 5)))
    return np.array(all_returns)


def problem3(config, iterations: int=200):
    """
    Repeat the previous question, but using the GA (as described earlier in 
    this assignment) on the More-Watery 687-Gridworld domain. Report the same 
    quantities.
    """
    all_returns = []

    def evaluate(p, episodes):
        returns = []
        for i in range(episodes):
            r = run_gridworld_episode(p)
            returns.append(r)
            all_returns.append(r)

        return np.mean(returns)

    agent_policy = TabularSoftmax(25, 4)
    agent = GA(populationSize=config[0], evaluationFunction=evaluate, initPopulationFunction=init_gridworld_population,
               numElite=config[1], numEpisodes=config[2], alpha=config[3], parent_frac=config[4])
    bar = range(iterations)
    for i in bar:
        agent_policy.parameters = agent.train()
        # bar.set_description("Average return: {}".format(evaluate(agent_policy.parameters, 5)))
    return np.array(all_returns)


def problem4(config, iterations: int=25):
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
        returns = []
        for i in range(episodes):
            r = run_cartpole_episode(p, config[0])
            returns.append(r)
            all_returns.append(r)

        return np.mean(returns)

    agent_policy = LinearApproximation(state_dim=4, num_actions=2, basis=config[0])
    agent = CEM(agent_policy.parameters, sigma=config[1], popSize=config[2], numElite=config[3], numEpisodes=config[4],
                evaluationFunction=evaluate, epsilon=config[5])
    for i in range(iterations):
        agent_policy.parameters = agent.train()

    return np.array(all_returns)


def problem5(config, iterations: int=100):
    """
    Repeat the previous question, but using first-choice hill-climbing (as 
    described in class) on the cart-pole domain. Report the same quantities 
    and how the policy was parameterized. 
    
    """
    all_returns = []

    def evaluate(p, episodes):
        returns = []
        for i in range(episodes):
            r = run_cartpole_episode(p, config[0])
            returns.append(r)
            all_returns.append(r)

        return np.mean(returns)

    agent_policy = LinearApproximation(state_dim=4, num_actions=2, basis=config[0])
    agent = FCHC(agent_policy.parameters, sigma=config[1], evaluationFunction=evaluate, numEpisodes=config[2])
    for i in range(iterations):
        agent_policy.parameters = agent.train()

    return np.array(all_returns)


def problem6(config, iterations: int=25):
    """
    Repeat the previous question, but using the GA (as described earlier in 
    this homework) on the cart-pole domain. Report the same quantities and how
    the policy was parameterized. 
    """
    all_returns = []

    def evaluate(p, episodes):
        returns = []
        for i in range(episodes):
            r = run_cartpole_episode(p, config[0])
            returns.append(r)
            all_returns.append(r)

        return np.mean(returns)

    agent_policy = LinearApproximation(state_dim=4, num_actions=2, basis=config[0])
    if config[0] == 2:
        agent = GA(populationSize=config[1], evaluationFunction=evaluate,
                   initPopulationFunction=init_cartpole_population_2, numElite=config[2], numEpisodes=config[3],
                   alpha=config[4], parent_frac=config[5])
    else:
        agent = GA(populationSize=config[1], evaluationFunction=evaluate,
                   initPopulationFunction=init_cartpole_population_3, numElite=config[2], numEpisodes=config[3],
                   alpha=config[4], parent_frac=config[5])
    for i in range(iterations):
        agent_policy.parameters = agent.train()

    return np.array(all_returns)


def run_config(params):

    def get_config_string(c):
        ret_str = ''
        for item in c:
            ret_str += ('_' + str(item))
        return ret_str

    config = params[3]
    trial_returns = []
    for trial in range(params[1]):
        trial_returns.append(params[0](config))
    trial_returns = np.array(trial_returns)

    means = trial_returns.mean(axis=0)
    errors = np.sqrt(trial_returns.var(axis=0))

    fig = plt.figure()
    plt.errorbar(range(len(means)), means, yerr=errors)
    plt.xlabel('Episode Count')
    plt.ylabel('Total reward')
    plt.savefig('plots/problem' + str(params[2]) + get_config_string(config) + '.png')
    plt.close(fig)


def main():

    sigma = [1, 2.5, 5]
    popSize = [10, 50]
    numElite = [1, 5]
    numEpisodes = [5, 25]
    epsilon = [0.1, 1.5, 3]
    alpha = [1, 2.5]
    parent_frac = [2, 3]
    basis = [2, 3]

    configs = {}
    configs[1] = [(s, p, e, ep, eps) for s in sigma for p in popSize for e in numElite for ep in numEpisodes
                  for eps in epsilon]
    configs[2] = [(s, ep) for s in sigma for ep in numEpisodes]
    configs[3] = [(p, e, ep, a, f) for p in popSize for e in numElite for ep in numEpisodes for a in alpha
                  for f in parent_frac]

    configs[4] = [(b, s, p, e, ep, eps) for s in sigma for p in popSize for e in numElite for ep in numEpisodes
                  for eps in epsilon for b in basis]
    configs[5] = [(b, s, ep) for s in sigma for ep in numEpisodes for b in basis]
    configs[6] = [(b, p, e, ep, a, f) for p in popSize for e in numElite for ep in numEpisodes for a in alpha
                  for f in parent_frac for b in basis]

    def run_experiment(problem, num_trials=50, problem_id=0):
        params = [[problem, num_trials, problem_id, c] for c in configs[problem_id]]
        pool = Pool(50)
        pool.map(run_config, params)

    print('Running experiment 1...')
    run_experiment(problem1, 50, 1)
    print('Running experiment 2...')
    run_experiment(problem2, 5, 2)
    print('Running experiment 3...')
    run_experiment(problem3, 50, 3)
    print('Running experiment 4...')
    run_experiment(problem4, 50, 4)
    print('Running experiment 5...')
    run_experiment(problem5, 5, 5)
    print('Running experiment 6...')
    run_experiment(problem6, 50, 6)


if __name__ == "__main__":
    main()
