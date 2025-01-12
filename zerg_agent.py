from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.agents import base_agent
import numpy as np


class ZergAI(base_agent.BaseAgent):
    def step(self, obs):
        super(ZergAI, self).step(obs)
        
        # Extract the available actions
        #available_actions = obs.observation.available_actions

        # # Build an Overlord if supply is close to the limit
        # if self.need_overlord(obs):
        #     return self.build_overlord(obs)

        # # Train a Drone if resources are sufficient
        # if self.can_train_drone(obs):
        #     return self.train_drone(obs)

        # # Spawn Zerglings from a Larva
        # if self.can_spawn_zerglings(obs):
        #     return self.spawn_zerglings(obs)

        # No operation if no action is taken
        return actions.FUNCTIONS.no_op()

    def need_overlord(self, obs):
        food_cap = obs.observation.player.food_cap
        food_used = obs.observation.player.food_used
        return food_cap - food_used < 2

    def build_overlord(self, obs):
        if actions.FUNCTIONS.Train_Overlord_quick.id in obs.observation.available_actions:
            return actions.FUNCTIONS.Train_Overlord_quick("now")
        return actions.FUNCTIONS.no_op()

    def can_train_drone(self, obs):
        if actions.FUNCTIONS.Train_Drone_quick.id in obs.observation.available_actions:
            return True
        return False

    def train_drone(self, obs):
        return actions.FUNCTIONS.Train_Drone_quick("now")

    def can_spawn_zerglings(self, obs):
        if actions.FUNCTIONS.Train_Zergling_quick.id in obs.observation.available_actions:
            return True
        return False

    def spawn_zerglings(self, obs):
        return actions.FUNCTIONS.Train_Zergling_quick("now")
