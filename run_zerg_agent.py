from pysc2.env import sc2_env
from pysc2.lib import features
from zerg_agent import ZergAI
from absl import app


def main(unused_argv):
    """
    @brief Initializes and runs a Zerg AI agent in a StarCraft II environment.

    This function sets up the environment for a Zerg agent to play against a Terran bot. It configures
    the SC2 environment, initializes the agent, and runs a simulation where the agent takes actions 
    based on the observations provided by the environment.

    @param unused_argv: The command-line arguments. This parameter is not used in the current implementation.
    """
    # Create a new instance of the ZergAI agent
    agent = ZergAI()

    try:
        # Set up the StarCraft II environment with a Zerg agent vs Terran bot
        with sc2_env.SC2Env(
            map_name="Simple64",  # Map used for the game
            players=[
                sc2_env.Agent(sc2_env.Race.zerg),  # The AI agent controlling the Zerg race
                sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.easy)  # Opponent bot (Terran)
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),  # Feature dimensions
                use_feature_units=True  # Use the feature units for unit-level information
            ),
            step_mul=8,  # Number of simulation steps per game step
            game_steps_per_episode=0,  # Number of steps per episode (0 means no limit)
            visualize=True  # Whether to visualize the game (True/False)
        ) as env:
            # Set up the agent with the environment's observation and action specs
            agent.setup(env.observation_spec(), env.action_spec())

            # Reset the environment and start the first episode
            timesteps = env.reset()
            agent.reset()

            # Run the simulation loop until the episode ends
            while True:
                # The agent decides on an action for this step
                step_actions = [agent.step(timesteps[0])]
                
                # If the episode is finished, break the loop
                if timesteps[0].last():
                    break
                
                # Step the environment with the agent's action and get the next timestep
                timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        # Gracefully handle a KeyboardInterrupt (Ctrl+C)
        pass


if __name__ == "__main__":
    """
    @brief Entry point for running the Zerg AI in the SC2 environment.

    This block of code runs the main function when the script is executed directly. It ensures the
    SC2 environment is initialized and that the agent is set up to start playing.

    @note This is the entry point of the program.
    """
    app.run(main)
