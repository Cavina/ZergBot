import random
import math

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
import zerg_definitions
import zerg_actions 
from pysc2.lib import actions
from pysc2.lib import features
from logger_config import logger

class SmartZergAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SmartZergAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(zerg_actions.smart_actions))))

    def transformLocation(self, x, x_one, y, y_one):
        if not self.base_top_left:
            return [x - x_one, y - y_one]
        
        return [x + x_one, y + y_one]
    

    def step(self, obs):
        super(SmartZergAgent, self).step(obs)
        #minimap should be feature_minimap
        player_y, player_x = (obs.observation['feature_screen'][zerg_definitions._PLAYER_RELATIVE] == zerg_definitions._PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        smart_action = zerg_actions.smart_actions[random.randrange(0, len(zerg_actions.smart_actions) - 1)]

        if smart_action == zerg_actions.ACTION_DO_NOTHING:
            return actions.FunctionCall(zerg_definitions._NO_OP, [])
        elif smart_action == zerg_actions.ACTION_SELECT_DRONE:
            unit_type = obs.observation['feature_screen'][zerg_definitions._UNIT_TYPE]
            unit_y, unit_x = (unit_type == zerg_definitions._ZERG_DRONE).nonzero()

            if unit_y.any():
                i = random.randint(0, len(unit_y)-1)
                target = [unit_x[i], unit_y[i]]

                return actions.FunctionCall(zerg_definitions._SELECT_POINT, [zerg_definitions._NOT_QUEUED, target])
        elif smart_action == zerg_actions.ACTION_SELECT_LARVA:
            unit_type = obs.observation['feature_screen'][zerg_definitions._UNIT_TYPE]
            unit_y, unit_x = (unit_type == zerg_definitions._ZERG_LARVA).nonzero()

            if unit_y.any():
                i = random.randint(0, len(unit_y)-1)
                target = [unit_x[i], unit_y[i]]
                return actions.FunctionCall(zerg_definitions._SELECT_POINT, [zerg_definitions._NOT_QUEUED, target])
        elif smart_action == zerg_actions.ACTION_BUILD_SPAWNINGPOOL:
            if zerg_definitions._BUILD_SPAWNINGPOOL in obs.observation['available_actions']:
                unit_type = obs.observation['feature_screen'][zerg_definitions._UNIT_TYPE]
                unit_y, unit_x = (unit_type == zerg_definitions._ZERG_HATCHERY).nonzero()

                if unit_y.any():
                    target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
                    return actions.FunctionCall(zerg_definitions._BUILD_SPAWNINGPOOL, [zerg_definitions._NOT_QUEUED, target])
        elif smart_action == zerg_actions.ACTION_BUILD_ZERGLING:
            if zerg_definitions._TRAIN_ZERGLING in obs.observation['available_actions']:
                return actions.FunctionCall(zerg_definitions._TRAIN_ZERGLING, [zerg_definitions._NOT_QUEUED])
        elif smart_action == zerg_actions.ACTION_BUILD_OVERLORD:
            if zerg_definitions._TRAIN_OVERLORD in obs.observation['available_actions']:
                supply_available = obs.observation["player"][4] - obs.observation["player"][3]
                if supply_available == 0:
                    return actions.FunctionCall(zerg_definitions._TRAIN_OVERLORD, [zerg_definitions._NOT_QUEUED])
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

        self.q_table.ix[s, a] += self.lr * (q_target-q_predict)

    def check_state_exists(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

