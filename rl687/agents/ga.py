import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class GA(BBOAgent):
    """
    A canonical Genetic Algorithm (GA) for policy search is a black box 
    optimization (BBO) algorithm. 
    
    Parameters
    ----------
    populationSize (int): the number of individuals per generation
    numEpisodes (int): the number of episodes to sample per policy         
    evaluationFunction (function): evaluates a parameterized policy
        input: a parameterized policy theta, numEpisodes
        output: the estimated return of the policy            
    numElite (int): the number of top individuals from the current generation
                    to be copied (unmodified) to the next generation
    """

    def __init__(self, populationSize:int, evaluationFunction:Callable, initPopulationFunction:Callable,
                 numElite:int=1, numEpisodes:int=5, alpha=2.5, parent_frac: int=2):
        self._name = 'Genetic_Algorithm'
        self._pop_size = populationSize
        self._num_elite = numElite
        self._num_parents = populationSize // parent_frac
        self._num_episodes = numEpisodes
        self._evaluation_function = evaluationFunction

        self._init_function = initPopulationFunction
        self._population = initPopulationFunction(populationSize)
        self._pop_evals = None
        self._num_children = populationSize - numElite
        self._alpha = alpha

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> np.ndarray:
        return self._pop_evals[0][0]

    def _mutate(self, parent: np.ndarray)->np.ndarray:
        """
        Perform a mutation operation to create a child for the next generation.
        The parent must remain unmodified. 
        
        output:
            child -- a mutated copy of the parent
        """
        return parent + self._alpha * np.random.normal(size=parent.shape)

    @staticmethod
    def get_parents(p, population):
        return population[:p]

    def get_children(self, parents):
        children = []
        for i in range(self._num_children):
            parent_id = np.random.choice(range(len(parents)))
            children.append(self._mutate(parents[parent_id]))

        return children

    def train(self)->np.ndarray:

        if self._pop_evals is None:
            samples = []
            for k in range(self._pop_size):
                samples.append((self._population[k], self._evaluation_function(self._population[k], self._num_episodes)))
            samples.sort(key=lambda x: x[1], reverse=True)
            self._pop_evals = samples

        else:
            samples = [sample[0] for sample in self._pop_evals]

            parents = self.get_parents(self._num_parents, samples)
            nex_gen = samples[:self._num_elite] + self.get_children(parents)
            self._population = np.array(nex_gen)

            samples = []
            for k in range(self._pop_size):
                samples.append(
                    (self._population[k], self._evaluation_function(self._population[k], self._num_episodes)))
            samples.sort(key=lambda x: x[1], reverse=True)
            self._pop_evals = samples

        return self._pop_evals[0][0]

    def reset(self)->None:
        self._population = self._init_function(self._pop_size)
