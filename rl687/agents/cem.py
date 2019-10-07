import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class CEM(BBOAgent):
    """
    The cross-entropy method (CEM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is based on Stulp and Sigaud (2012). Intuitively,
    CEM starts with a multivariate Gaussian dsitribution over policy parameter vectors.
    This distribution has mean thet and covariance matrix Sigma. It then samples some
    fixed number, K, of policy parameter vectors from this distribution. It evaluates
    these K sampled policies by running each one for N episodes and averaging the
    resulting returns. It then picks the K_e best performing policy parameter
    vectors and fits a multivariate Gaussian to these parameter vectors. The mean and
    covariance matrix for this fit are stored in theta and Sigma and this process
    is repeated.

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """

    def __init__(self, theta: np.ndarray, sigma: float, popSize: int, numElite: int, numEpisodes: int,
                 evaluationFunction: Callable, epsilon: float=1.5):
        self.initial_params = [theta, sigma * np.eye(len(theta))]
        self._name = 'Cross_Entropy_Method'
        self._theta = theta
        self._Sigma = sigma * np.eye(len(theta))
        self._pop_size = popSize
        self._num_elite = numElite
        self._num_episodes = numEpisodes
        self._evaluation_function = evaluationFunction
        self._epsilon = epsilon

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        return self._theta

    def train(self)->np.ndarray:
        samples = []
        for k in range(self._pop_size):
            theta_sample = np.random.multivariate_normal(self._theta, self._Sigma)
            j = self._evaluation_function(theta_sample, self._num_episodes)
            samples.append((theta_sample, j))
        samples.sort(key=lambda x: x[1], reverse=True)
        elite_samples = np.array([sample[0] for sample in samples[:self._num_elite]])
        elite_mean = elite_samples.mean(axis=0)
        elite_cov = (np.einsum('ij,ki->ijk',
                               (elite_samples - elite_mean), (elite_samples - elite_mean).T).sum(axis=0) +
                     self._epsilon * np.eye(len(self._theta))) / (self._epsilon + self._num_elite)

        self._theta = elite_mean
        self._Sigma = elite_cov

        return self._theta

    def reset(self)->None:
        self._theta = self.initial_params[0]
        self._Sigma = self.initial_params[1]
