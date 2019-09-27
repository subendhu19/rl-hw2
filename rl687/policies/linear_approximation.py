import numpy as np
from .skeleton import Policy
from typing import Union
from scipy.special import softmax
import math


def number_to_base(n, b, md):
    if n == 0:
        return [0] * md
    digits = []
    while n > 0:
        digits.append(int(n % b))
        n //= b
    return [0]*(max(0, md-len(digits))) + digits[::-1]


class LinearApproximation(Policy):
    """
    Softmax action selection using a linear function approximation

    Parameters
    ----------
    State vector size (int): dimenstions of the state vector
    numActions (int): the number of actions the tabular softmax policy has
    """

    def __init__(self, state_dim: int, numActions: int):
        self._basis = 3
        self._state_dim = state_dim
        self._num_actions = numActions
        self._feature_dim = (self._basis + 1) ** self._state_dim
        self._theta = np.random.normal(size=(numActions, self._feature_dim))

    @property
    def parameters(self) -> np.ndarray:
        return self._theta.flatten()

    @parameters.setter
    def parameters(self, p: np.ndarray):
        self._theta = p.reshape(self._theta.shape)

    def __call__(self, state: np.ndarray, action=None) -> Union[float, np.ndarray]:
        if action:
            return self.getActionProbabilities(state)[action]
        else:
            return self.getActionProbabilities(state)

    def samplAction(self, state: np.ndarray) -> int:
        action_probs = self.getActionProbabilities(state)
        return np.random.choice(np.arange(self._num_actions), p=action_probs)

    def getFourierTransform(self, state: np.ndarray) -> np.ndarray:
        cv = [number_to_base(i, self._basis + 1, self._state_dim) for i in range(self._feature_dim)]
        return np.array([math.cos(i * math.pi * (np.dot(state, cv[i]))) for i in range(self._feature_dim)])

    def getActionProbabilities(self, state: np.ndarray) -> np.ndarray:
        action_logits = np.dot(self._theta, self.getFourierTransform(state))
        action_probs = softmax(action_logits)
        return action_probs
