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

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5

class SmartZergAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SmartZergAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(zerg_actions.smart_actions))))

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

        self.previous_action = None
        self.previous_state = None

    def transformDistance(self, x, x_one, y, y_one):
        if not self.base_top_left:
            return [x - x_one, y - y_one]
        
        return [x + x_one, y + y_one]
    
    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64-x, 64-y]
        
        return [x,y]
    



    def step(self, obs):
        super(SmartZergAgent, self).step(obs)
        #minimap should be feature_minimap
        player_y, player_x = (obs.observation['feature_screen'][zerg_definitions._PLAYER_RELATIVE] == zerg_definitions._PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        unit_type = obs.observation['feature_screen'][zerg_definitions._UNIT_TYPE]

        overlord_y, overlord_x = (unit_type == zerg_definitions._ZERG_OVERLORD).nonzero()
        overlord_count = 1 if overlord_y.any else 0

        spawningpool_y, spawningpool_x = (unit_type == zerg_definitions._ZERG_SPAWNINGPOOL).nonzero()
        spawningpool_count = 1 if spawningpool_y.any() else 0

        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]

        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]

        


        current_state = [
            overlord_count,
            spawningpool_count,
            supply_limit,
            army_supply
        ]

        if self.previous_action is not None:
            reward = 0

            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD
            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = zerg_actions.smart_actions[rl_action]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action




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
                    target = self.transformDistance(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
                    return actions.FunctionCall(zerg_definitions._BUILD_SPAWNINGPOOL, [zerg_definitions._NOT_QUEUED, target])
        elif smart_action == zerg_actions.ACTION_BUILD_ZERGLING:
            if zerg_definitions._TRAIN_ZERGLING in obs.observation['available_actions']:
                return actions.FunctionCall(zerg_definitions._TRAIN_ZERGLING, [zerg_definitions._NOT_QUEUED])
        
        elif smart_action == zerg_actions.ACTION_BUILD_OVERLORD:
            if zerg_definitions._TRAIN_OVERLORD in obs.observation['available_actions']:
                supply_available = obs.observation["player"][4] - obs.observation["player"][3]
                if supply_available == 0:
                    return actions.FunctionCall(zerg_definitions._TRAIN_OVERLORD, [zerg_definitions._NOT_QUEUED])
                
        elif smart_action == zerg_actions.ACTION_SELECT_ARMY:
            if zerg_definitions._SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(zerg_definitions._SELECT_ARMY, [zerg_definitions._NOT_QUEUED])

        elif smart_action == zerg_actions.ACTION_ATTACK:
            if zerg_definitions._ATTACK_MINIMAP in obs.observation['available_actions']:
                if self.base_top_left:
                    return actions.FunctionCall(zerg_definitions._ATTACK_MINIMAP, [zerg_definitions._NOT_QUEUED, [39, 45]])
                return actions.FunctionCall(zerg_definitions._ATTACK_MINIMAP, [zerg_definitions._NOT_QUEUED, [21, 24]])
        return actions.FunctionCall(zerg_definitions._NO_OP, [])

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


    def choose_action(self, observation):
        self.check_state_exists(observation)

        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]

            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action
    

    def learn(self, s, a, r, s_):
        self.check_state_exists(s_)
        self.check_state_exists(s)

        q_predict = self.q_table.loc[s, a]
        q_target = r + self.gamma * self.q_table.loc[s_, :].max()

        self.q_table.loc[s, a] += self.lr * (q_target-q_predict)

    def check_state_exists(self, state):
        if state not in self.q_table.index:
            new_row = pd.DataFrame([[0] * len(self.actions)], columns=self.q_table.columns, index=[state])
            self.q_table = pd.concat([self.q_table, new_row])
