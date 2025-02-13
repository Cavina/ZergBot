import logging

# Create a general logger
logger = logging.getLogger("ZergAI")
logger.setLevel(logging.DEBUG)

# Create an RL actions-specific logger
rl_logger = logging.getLogger("RL_Actions")
rl_logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = logging.FileHandler("debug.log")
file_handler.setLevel(logging.DEBUG)

rl_file_handler = logging.FileHandler("rl_actions.log")  # New log file
rl_file_handler.setLevel(logging.DEBUG)

# Create formatters
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
rl_file_handler.setFormatter(formatter)

# Add handlers to each logger
logger.addHandler(file_handler)       # General logging
rl_logger.addHandler(rl_file_handler)  # RL action-specific logging
