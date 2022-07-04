"""
Reinforcement learning agent
"""
import numpy as np

from .observation import Observer
from rl.models.dddqn import Policy
from train_sim.train import Train
from train_sim.constants import *


class Agent:
    def __init__(self,
                 train: Train,
                 observer: Observer,
                 rng: np.random.Generator):
        self.train: Train = train
        self.train.set_agent(self)
        self.observer = observer
        self.simulation = train.simulation
        self.rng = rng

    def get_observation(self):
        """
        Get the train observation
        :return: Get the train tree observation
        """
        return self.observer.get_observation()

    def get_state_and_actions(self):
        """
        Get the actions of the train based on the agent models
        :return: A vector of 4 floating point number between 0 to 1 representing the actions to be taken in the order of
            MOVE_FORWARD, REVERSE, TURN_LEFT, TURN_RIGHT
        """
        return self.get_observation(), self.rng.choice([TRAIN_ACTION_TURN_LEFT, TRAIN_ACTION_TURN_RIGHT])


class DefaultAgent(Agent):
    def get_state_and_actions(self):
        """
        Get the actions of the train based on the agent models
        :return: A vector of 4 floating point number between 0 to 1 representing the actions to be taken in the order of
            MOVE_FORWARD, REVERSE, TURN_LEFT, TURN_RIGHT
        """
        return self.get_observation(), self.rng.choice([TRAIN_ACTION_TURN_LEFT, TRAIN_ACTION_TURN_RIGHT])


class RandomAgent(Agent):
    def __init__(self,
                 train: Train,
                 observer: Observer,
                 rng: np.random.Generator):
        super().__init__(train, observer, rng)

    def get_state_and_actions(self):
        """
        Get the actions of the train based on the random agent models, which will basically performs random action
        :return: A vector of 4 floating point number between 0 to 1 representing the actions to be taken in the order of
            MOVE_FORWARD, REVERSE, TURN_LEFT, TURN_RIGHT
        """
        return self.get_observation(), self.rng.choice([
            TRAIN_ACTION_BREAK, TRAIN_ACTION_REVERSE, TRAIN_ACTION_TURN_LEFT, TRAIN_ACTION_TURN_RIGHT])


class PolicyAgent(Agent):
    def __init__(self,
                 train: Train,
                 observer: Observer,
                 policy: Policy,
                 eps: float,
                 rng: np.random.Generator):
        """

        :param train: The train in which the agent controls
        :param observer: Observer of this policy
        :param policy: The models that dictates the action policy
        :param eps: Exploration probability
        :param rng: Input RNG, used for determinism
        """
        super().__init__(train, observer, rng)
        self.policy = policy
        self.eps = eps

    def get_state_and_actions(self):
        """
        Get the actions of the train based on the given policy models, which will basically performs random action
        :return: A vector of 4 floating point number between 0 to 1 representing the actions to be taken in the order of
            MOVE_FORWARD, REVERSE, TURN_LEFT, TURN_RIGHT
        """
        obs = self.get_observation()
        action = self.policy.act(obs, self.eps)
        return obs, action
