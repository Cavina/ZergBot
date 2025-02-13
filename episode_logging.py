import logging
import pandas as pd
import os

# Define log file names
EPISODE_LOG_FILE = "episode_log.log"
CSV_LOG_FILE = "rl_episode_log.csv"

# Create a logger for episodic data
episode_logger = logging.getLogger("EpisodicLogger")
episode_logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler(EPISODE_LOG_FILE)
file_handler.setLevel(logging.DEBUG)

# Define log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add handler to logger
episode_logger.addHandler(file_handler)


# Function to log episodic stats
def log_episode(episode, total_reward, steps, apm, units_killed):
    """
    Logs episodic performance to both a log file and a CSV file.
    """
    if total_reward == 1:
        result_str = "Win"
    if total_reward == -1:
        result_str = "Loss"
    else:
        result_str = "Tie"
    # Log to log file
    episode_logger.info(
        f"Episode: {episode} | Result: {result_str} | Reward: {total_reward} | "
        f"Steps: {steps} | APM: {apm} | Units Killed: {units_killed}"
    )

    # Log to CSV
    log_data = {
        "Episode": [episode],
        "Win": [result_str],
        "Total Reward": [total_reward],
        "Steps": [steps],
        "APM": [apm],
        "Units Killed": [units_killed],
    }

    df = pd.DataFrame(log_data)

    # Append data to CSV file
    if not os.path.exists(CSV_LOG_FILE):  
        df.to_csv(CSV_LOG_FILE, index=False)  # Create a new file if it doesn't exist
    else:
        df.to_csv(CSV_LOG_FILE, mode="a", header=False, index=False)  # Append to existing file
