import abc
from typing import Tuple


class env:
    def __init__(self):
        self.pre_action = None

    @abc.abstractmethod
    def check(self) -> Tuple:
        """
        check if the game is over.
        if done == 0 means the game is not complete, else means the game is over and the winner is as done.
        using pre_action to check, and this function will be called after step().
        """
        pass

    @abc.abstractmethod
    def step(self, agent, action):
        """
        step the game given (agent, action)
        !!this function must push the action to pre_action.
        """
        pass

    @abc.abstractmethod
    def get_state(self,agent):
        """
        such as: return state.copy()
        """
        pass

    @abc.abstractmethod
    def get_reward(self):
        """
        return reward of the pre_action.
        """
        pass
