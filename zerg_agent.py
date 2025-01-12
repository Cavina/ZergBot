from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.agents import base_agent
import numpy as np
import random
from logger_config import logger

class ZergAI(base_agent.BaseAgent):
    """
    @brief An AI Agent for the Zerg Race.
    
    This class defines an AI agent for controlling Zerg units in StarCraft II. It inherits from
    the BaseAgent class and implements the `step` function to define the agent's actions in each step.
    """
    def __init__(self):
        super(ZergAI, self).__init__()

        self.attack_coordinates = None

    def step(self, obs):
        """
        @brief Decides the next action for the Zerg AI.
        
        This function is called at each step of the game. It selects a drone unit and issues a command
        to select all drones. If no drones are available, it returns a no-op action.
        
        @param obs: The observation data from the environment.
        @return: The action that the agent decides to take (either selecting a point or a no-op).
        """
       
        super(ZergAI, self).step(obs)

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            xmean = player_x.mean()
            ymean = player_y.mean()


            if xmean <= 31 and ymean <= 31:
                self.attack_coordinates = (49, 49)
            else:
                self.attack_coordinates = (12, 16)            
        zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
        if len(zerglings) >= 10:
            if self.unit_type_is_selected(obs, units.Zerg.Zergling):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap("now",
                                                  self.attack_coordinates)

            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")
            
        spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
        
        if len(spawning_pools) == 0:
            if self.unit_type_is_selected(obs, units.Zerg.Drone):
                if (actions.FUNCTIONS.Build_SpawningPool_screen.id in
                    obs.observation.available_actions):
                    x = random.randint(0, 83)
                    y = random.randint(0, 83)

                    return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x,y))

        # Collect all drones from the observation
            drones = [unit for unit in obs.observation.feature_units
                if unit.unit_type == units.Zerg.Drone]
        
        # If drones exist, select one at random
            if len(drones) > 0:
                drone = random.choice(drones)

            # Return a command to select the chosen drone's position
            return actions.FUNCTIONS.select_point("select_all_type", (drone.x, drone.y))
        
        if self.unit_type_is_selected(obs, units.Zerg.Larva):
            free_supply = (obs.observation.player.food_cap -
                obs.observation.player.food_used)
            if free_supply == 0:
                if self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
                    return actions.FUNCTIONS.Train_Overlord_quick("now")

            if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                return actions.FUNCTIONS.Train_Zergling_quick("now")
    
        larvae = self.get_units_by_type(obs, units.Zerg.Larva)
        if len(larvae) > 0:
            larva = random.choice(larvae)
      
            return actions.FUNCTIONS.select_point("select_all_type", (larva.x,
                                                                larva.y))

        # Return a no-op action if no drones are available
        return actions.FUNCTIONS.no_op()

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