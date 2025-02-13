import pandas as pd
import json
import matplotlib.pyplot as plt

# Load game stats from JSON
with open("game_stats.json", "r") as f:
    stats = json.load(f)

# Convert to Pandas DataFrame
df = pd.DataFrame([stats])

# Ensure ties are calculated
df["ties"] = df["games_played"] - df["wins"] - df["losses"]

# Extract total games played as an integer
games_played = int(df["games_played"].iloc[0])

# Generate X-axis values (Game Numbers)
games = range(1, games_played + 1)

# Plot Wins, Losses, and Ties Over Time
plt.plot(games, [df["wins"].iloc[0]] * games_played, label="Wins", linestyle='-', marker='o')
plt.plot(games, [df["losses"].iloc[0]] * games_played, label="Losses", linestyle='-', marker='x')
plt.plot(games, [df["ties"].iloc[0]] * games_played, label="Ties", linestyle='-', marker='s')

# Labels and Title
plt.xlabel("Games Played")
plt.ylabel("Count")
plt.title("Win/Loss/Tie Progression")
plt.legend()

# Show the plot
plt.show()
