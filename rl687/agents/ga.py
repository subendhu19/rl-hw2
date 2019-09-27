import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable
from tqdm import trange


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

    def __init__(self, populationSize:int, evaluationFunction:Callable, numElite:int=10, numEpisodes:int=25,
                 num_params: int=100):
        self._name = 'Genetic algorithm'
        self._pop_size = populationSize
        self._num_elite = numElite
        self._num_parents = 5
        self._num_episodes = numEpisodes
        self._evaluation_function = evaluationFunction

        self._theta = np.random.normal(size=num_params)
        self._num_params = num_params
        self._num_generations = 300
        self._num_children = populationSize - numElite
        self._alpha = 2.5

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> np.ndarray:
        return self._theta

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
            parent_id = np.random.choice(range(self._num_parents))
            children.append(self._mutate(parents[parent_id]))

        return children

    def train(self)->np.ndarray:
        initial_population = []
        for i in range(self._pop_size):
            initial_population.append(np.random.multivariate_normal(self._theta, np.eye(self._theta.shape[0])))

        population = initial_population

        print('Training the {} agent...'.format(self.name))
        bar = trange(self._num_generations)
        for generation in bar:
            samples = []
            for k in range(self._pop_size):
                samples.append((population[k], self._evaluation_function(population[k], self._num_episodes)))
            samples.sort(key=lambda x: x[1], reverse=True)

            samples = [sample[0] for sample in samples]

            parents = self.get_parents(self._num_parents, samples)
            nex_gen = samples[:self._num_elite] + self.get_children(parents)
            population = nex_gen
            self._theta = np.array(population).mean(axis=0)
            bar.set_description("Average return: {}".format(self._evaluation_function(self._theta, self._num_episodes)))

        print('Finished training')

        return self._theta

    def reset(self)->None:
        self._theta = np.random.normal(size=self._num_params)
