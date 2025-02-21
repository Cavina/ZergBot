import random
import math
import os

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
import zerg_definitions
import zerg_actions 
from pysc2.lib import actions
from pysc2.lib import features
from logger_config import logger, rl_logger
import game_stats_tracker as gs
from q_learning_table import *
from episode_logging import *
# KILL_UNIT_REWARD = 0.2
# KILL_BUILDING_REWARD = 0.5

DATA_FILE = 'sparse_agent_data'
TOTAL_ACTIONS = 0

'''
Narrows the search space down to a 4x4 grid

'''
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            zerg_actions.smart_actions.append(zerg_actions.ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))

class SmartZergAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SmartZergAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(zerg_actions.smart_actions))))
        self.wins, self.losses, self.ties, self.games_played = gs.load_game_stats() 
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        
        self.total_friendly_units_lost = 0
        self.prev_friendly_units = set()
        

        self.previous_action = None
        self.previous_state = None

        self.cc_y = None
        self.cc_x = None

        self.move_number = 0


        #Checks if there is an existing q-table and loads it.
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    #Transforms distance
    def transformDistance(self, x, x_one, y, y_one):
        if not self.base_top_left:
            return [x - x_one, y - y_one]
        
        return [x + x_one, y + y_one]
    
    #Transforms location 
    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64-x, 64-y]
        
        return [x,y]
    
    #Splits actions for future decision chains
    def splitAction(self, action_id):
        smart_action = zerg_actions.smart_actions[action_id]

        x, y = 0, 0

        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)



    #Step function for the environment
    def step(self, obs):
        super(SmartZergAgent, self).step(obs)
        #minimap should be feature_minimap
        #If we are on the last observation, we should learn and write the q-table as well as the game stats log
        global TOTAL_ACTIONS
        TOTAL_ACTIONS += 1
        game_time_seconds = obs.observation["game_loop"] / 22.4
        apm = (TOTAL_ACTIONS / game_time_seconds) * 60 if game_time_seconds > 0 else 0

        
        if obs.last():
            reward = obs.reward
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
            if reward > 0:
                self.wins += 1
            elif reward < 0:
                self.losses += 1
            else:
                self.ties += 1

            gs.save_game_stats(self.wins, self.losses, self.ties, self.games_played+1)


            # Update the stored unit tags for the next step
            # lost_units = self.prev_friendly_units - friendly_units
            # self.total_friendly_units_lost += len(lost_units)
            # self.prev_friendly_units = friendly_units
           
            log_episode(self.games_played+1, reward, obs.observation["game_loop"], apm, 0, self.total_friendly_units_lost)
            
            self.previous_action = None
            self.previous_state = None
            
            self.move_number = 0


            
            return actions.FunctionCall(zerg_definitions._NO_OP, [])



        unit_type = obs.observation['feature_screen'][zerg_definitions._UNIT_TYPE]
        #If the first observation, we can do some setup 
        if obs.first():
            player_y, player_x = (obs.observation['feature_minimap'][zerg_definitions._PLAYER_RELATIVE] == zerg_definitions._PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
            self.cc_y, self.cc_x = (unit_type == zerg_definitions._ZERG_HATCHERY).nonzero()
            
        #Checking for overlord counts
        overlord_y, overlord_x = (unit_type == zerg_definitions._ZERG_OVERLORD).nonzero()
        overlord_count = 1 if overlord_y.any else 0

        #Checking for spawning Pool counts
        spawningpool_y, spawningpool_x = (unit_type == zerg_definitions._ZERG_SPAWNINGPOOL).nonzero()
        spawningpool_count = 1 if spawningpool_y.any() else 0

        #Establishing supply_limits and current supply for future calculations
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]

        # killed_unit_score = obs.observation['score_cumulative'][5]
        # killed_building_score = obs.observation['score_cumulative'][6]

                
        #Store the initial states in the current state.
        current_state = np.zeros(20)
        current_state[0] = overlord_count
        current_state[1] = spawningpool_count
        current_state[2] = supply_limit
        current_state[3] = army_supply

        #Adding hot squares to store enemy positions.
        hot_squares = np.zeros(16)
        enemy_y, enemy_x = (obs.observation['feature_minimap'][zerg_definitions._PLAYER_RELATIVE] == zerg_definitions._PLAYER_HOSTILE).nonzero()

        #Fixed the previous error by constraining earlier and reducing the problem space size.
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))


            hot_squares[((y-1)*4) + (x-1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]
        
        for i in range(0, 4):
            current_state[i+4] = hot_squares[i]

        if self.previous_action is not None:
            # reward = 0

            # # if killed_unit_score > self.previous_killed_unit_score:
            # #     reward += KILL_UNIT_REWARD
            # # if killed_building_score > self.previous_killed_building_score:
            # #     reward += KILL_BUILDING_REWARD
            
            #Learn from every step so long as it is not the first step.
            self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

        #Extract the action index determined from the learning table
        rl_action = self.qlearn.choose_action(str(current_state))

        #Set the action to be taken
        smart_action = zerg_actions.smart_actions[rl_action]
        #Log the current state and smart action taken.
        #rl_logger.debug(str(current_state) + " smart_action: " + smart_action)

        # self.previous_killed_unit_score = killed_unit_score
        # # self.previous_killed_building_score = killed_building_score

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
        elif smart_action == zerg_actions.ACTION_BUILD_DRONE:
            if zerg_definitions._TRAIN_DRONE in obs.observation['available_actions']:
                return actions.FunctionCall(zerg_definitions._TRAIN_DRONE, [zerg_definitions._NOT_QUEUED])
        elif smart_action == zerg_actions.ACTION_BUILD_OVERLORD:
            if zerg_definitions._TRAIN_OVERLORD in obs.observation['available_actions']:
                supply_available = obs.observation["player"][4] - obs.observation["player"][3]
                if supply_available == 0:
                    return actions.FunctionCall(zerg_definitions._TRAIN_OVERLORD, [zerg_definitions._NOT_QUEUED])
                
        elif smart_action == zerg_actions.ACTION_SELECT_ARMY:
            if zerg_definitions._SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(zerg_definitions._SELECT_ARMY, [zerg_definitions._NOT_QUEUED])
        elif smart_action == zerg_actions.ACTION_ATTACK:
            unit_type = obs.observation['feature_screen'][zerg_definitions._UNIT_TYPE]
            unit_y, unit_x = (unit_type == zerg_definitions._ZERG_DRONE).nonzero() 

            if unit_y.any():
                return actions.FunctionCall(zerg_definitions._NO_OP, [])
            if zerg_definitions._ATTACK_MINIMAP in obs.observation['available_actions']:
                if self.base_top_left:
                    return actions.FunctionCall(zerg_definitions._ATTACK_MINIMAP, [zerg_definitions._NOT_QUEUED, [39, 45]])
                return actions.FunctionCall(zerg_definitions._ATTACK_MINIMAP, [zerg_definitions._NOT_QUEUED, [21, 24]])
            


     
        return actions.FunctionCall(zerg_definitions._NO_OP, [])



