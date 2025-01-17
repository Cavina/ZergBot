from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.agents import base_agent
import numpy as np
import random
from logger_config import logger
from behavior_tree import *

class ZergAI(base_agent.BaseAgent):
    """
    @brief An AI Agent for the Zerg Race.
    
    This class defines an AI agent for controlling Zerg units in StarCraft II. It inherits from
    the BaseAgent class and implements the `step` function to define the agent's actions in each step.
    """
    def __init__(self):
        super(ZergAI, self).__init__()

        self.attack_coordinates = None
        self.behavior_tree = self.create_behavior_tree()

    def create_behavior_tree(self):
        """Creates the behavior tree for the Zerg AI."""
        return Selector([
                ActionNode("Select Drone", self.action_select_drone),
            ActionNode("No Action", lambda obs: actions.FUNCTIONS.no_op())
        ])
      # Behavior Tree Conditions and Actions
    def check_needs_spawning_pool(self, obs):
        """Checks if a Spawning Pool is needed."""
        if self.needs_spawning_pool(obs):
            return "SUCCESS"
        return "FAILURE"

    def action_select_drone(self, obs):
        """Selects a drone for construction tasks."""
        action = self.select_drone(obs)
        if action:
            return action
        return "FAILURE"

    def action_build_spawning_pool(self, obs):
        """Builds a Spawning Pool if possible."""
        action = self.build_spawning_pool(obs)
        if action:
            return action
        return "FAILURE"

    def action_train_zerglings(self, obs):
        """Trains Zerglings if there are available larvae."""
        if self.has_available_larvae(obs):
            action = self.train_zerglings(obs)
            if action:
                return action
        return "FAILURE"   

    def step(self, obs):
        """
        @brief Decides the next action for the Zerg AI.
        
        This function is called at each step of the game. It selects a drone unit and issues a command
        to select all drones. If no drones are available, it returns a no-op action.
        
        @param obs: The observation data from the environment.
        @return: The action that the agent decides to take (either selecting a point or a no-op).
        """
       
        super(ZergAI, self).step(obs)
        action = self.behavior_tree.run(obs)
        #action = ActionNode("Select Drone", self.action_select_drone).run(obs)
        
        return action
        #return actions.FUNCTIONS.no_op()
 # Conditions
    def has_available_drones(self, obs):
        return len(self.get_units_by_type(obs, units.Zerg.Drone)) > 0
    
    def needs_spawning_pool(self, obs):
        return len(self.get_units_by_type(obs, units.Zerg.SpawningPool)) == 0

    def has_available_larvae(self, obs):
        larvae = self.get_units_by_type(obs, units.Zerg.Larva)
        return len(larvae) > 0
    
    def select_drone(self, obs):
        logger.debug("Entering select drone")
        drones = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Zerg.Drone]
        if drones:
            drone = random.choice(drones)
            logger.debug("Returning a drone")
            return actions.FUNCTIONS.select_point("select_all_type", (drone.x, drone.y))
        return None

    def build_spawning_pool(self, obs):
        logger.debug("Entering build_spawning pool")
        if self.unit_type_is_selected(obs, units.Zerg.Drone):
            logger.debug("Select DRone in build_spawning_pool confirmed")
            if actions.FUNCTIONS.Build_SpawningPool_screen.id in obs.observation.available_actions:
                logger.debug("Build Spawning Pool was in available actions and we should return properly")
                x = random.randint(0, 83)
                y = random.randint(0, 83)
                return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y))
        return None

    def train_zerglings(self, obs):
        larvae = self.get_units_by_type(obs, units.Zerg.Larva)
        if larvae:
            larva = random.choice(larvae)
            return actions.FUNCTIONS.select_point("select_all_type", (larva.x, larva.y))
        return None
    def unit_type_is_selected(self, obs, unit_type):
        """
        @brief Checks if a specific unit type is currently selected.

        This function determines whether the given unit type is currently selected in either the 
        single-select or multi-select group of the game observation.

        @param obs The game observation, which contains details about the current state of the game,            
        including selected units.

        @param unit_type The unit type to check for selection.

        @return True if the specified unit type is selected; False otherwise.
        """
        # Check if the unit type is selected in the single-select group
        if (len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
            return True
 
        # Check if the unit type is selected in the multi-select group
        if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
            return True

        # Return False if the unit type is not selected
        return False

    def get_units_by_type(self, obs, unit_type):
        """
        @brief Retrieves all units of a specific type from the game observation.

        This function filters the units in the current game observation and returns a list of all 
        units that match the specified unit type.

        @param obs The game observation, which contains details about the current state of the game, 
                including all visible units.
        @param unit_type The unit type to filter for in the observation.

        @return A list of units that match the specified unit type.
        """
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]
    
    def can_do(self, obs, action):
        return action in obs.observation.available_actions