import logging


# Create a custom logger
logger = logging.getLogger("ZergAI")
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = logging.FileHandler("debug.log")
file_handler.setLevel(logging.DEBUG)

# Create formatters and add them to the handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
