import random
import math

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
import zerg_definitions
from pysc2.lib import actions
from pysc2.lib import features
from logger_config import logger

class SmartZergAgent(base_agent.BaseAgent):
    def transformLocation(self, x, x_one, y, y_one):
        if not self.base_top_left:
            return [x - x_one, y - y_one]
        
        return [x + x_one, y + y_one]
    

    def step(self, obs):
        super(SmartZergAgent, self).step(obs)
        #minimap should be feature_minimap
        player_y, player_x = (obs.observation['feature_screen'][zerg_definitions._PLAYER_RELATIVE] == zerg_definitions._PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        return actions.FunctionCall(zerg_definitions._NO_OP, [])

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.ix[observation, :]

            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idmax()
        else:
            action = np.random.choice(self.actions)
        return action
    

    def learn(self, s, a, r, s_):
        self.check_state_exists(s_)
        self.check_state_exists(s)

        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        self.q_table.ix[s, a] += self.lr * (q_targtet-q_predict)

    def check_state_exists(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

