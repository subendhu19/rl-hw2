import numpy as np
from typing import Tuple
from .skeleton import Environment
import math


class Cartpole(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.

    Actions: left (0) and right (1)
    Reward: 1 always

    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    """

    def __init__(self):
        self._name = "cartpole"
        self._action = 0
        self._reward = 0
        self._isEnd = False
        self._gamma = 1

        # define the state # NOTE: you must use these variable names
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # timestep
        self._t = 0.0  # total time elapsed  NOTE: USE must use this variable

        # action set
        self.f_max = 10
        self.action_set = {0: -1 * self.f_max, 1: self.f_max}

        # termination variables
        self.fail_angle = math.pi / 2
        self.cart_boundary = 3
        self.time_limit = 20.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def action(self) -> int:
        return self._action

    @property
    def isEnd(self) -> bool:
        return self._isEnd

    @property
    def state(self) -> np.ndarray:
        return np.array([self._x, self._v, self._theta, self._dtheta])

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        force = self.action_set[action]

        dx = state[1]
        dtheta = state[3]
        ddtheta = (((self._g * math.sin(state[2])) + (math.cos(state[2]) * (-force - (self._mp * self._l * (state[3] ** 2) * math.sin(state[2])))
                                                      / (self._mc + self._mp))) / (self._l * ((4.0/3.0) - (self._mp * ((math.cos(state[2])) ** 2) / (self._mc + self._mp)))))
        dv = ((force + (self._mp * self._l * ((state[3] ** 2) * math.sin(state[2]) - ddtheta * math.cos(state[2]))))
              / (self._mc + self._mp))

        dstate = np.array([dx, dv, dtheta, ddtheta])
        new_state = state + self._dt * dstate

        return new_state

    def R(self, state: np.ndarray, action: int, nextState: np.ndarray) -> float:
        if not self.isEnd:
            return 1
        else:
            return 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        old_state = self.state
        new_state = self.nextState(old_state, action)

        self._t += self._dt
        [self._x, self._v, self._theta, self._dtheta] = new_state
        self._isEnd = self.terminal()

        self._reward = self.R(old_state, action, new_state)

        return self.state, self.reward, self.isEnd

    def reset(self) -> None:
        """
        resets the state of the environment to the initial configuration
        """
        self._x = 0.
        self._v = 0.
        self._theta = 0.
        self._dtheta = 0.
        self._t = 0.0

    def terminal(self) -> bool:
        """
        The episode is at an end if:
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """

        if abs(self.state[0]) > self.cart_boundary or abs(self.state[2]) > math.pi/12 or self._t > self.time_limit:
            return True

        return False
