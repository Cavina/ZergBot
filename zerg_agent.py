from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.agents import base_agent
import numpy as np
import random

class ZergAI(base_agent.BaseAgent):
    """
    @brief An AI Agent for the Zerg Race.
    
    This class defines an AI agent for controlling Zerg units in StarCraft II. It inherits from
    the BaseAgent class and implements the `step` function to define the agent's actions in each step.
    """

    def step(self, obs):
        """
        @brief Decides the next action for the Zerg AI.
        
        This function is called at each step of the game. It selects a drone unit and issues a command
        to select all drones. If no drones are available, it returns a no-op action.
        
        @param obs: The observation data from the environment.
        @return: The action that the agent decides to take (either selecting a point or a no-op).
        """
        
        super(ZergAI, self).step(obs)

        # Collect all drones from the observation
        drones = [unit for unit in obs.observation.feature_units
                  if unit.unit_type == units.Zerg.Drone]
        
        # If drones exist, select one at random
        if len(drones) > 0:
            drone = random.choice(drones)

            # Return a command to select the chosen drone's position
            return actions.FUNCTIONS.select_point("select_all_type", (drone.x, drone.y))
        
        # Return a no-op action if no drones are available
        return actions.FUNCTIONS.no_op()
