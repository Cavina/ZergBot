import os
import json

STATS_FILE = "game_stats.json"

def save_game_stats(wins, losses, games_played):
    """Saves game stats to a JSON file."""
    stats = {
        "wins": wins,
        "losses": losses,
        "games_played": games_played
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)

def load_game_stats():
    """Loads game stats from a JSON file."""
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            stats = json.load(f)
            return stats.get("wins", 0), stats.get("losses", 0), stats.get("games_played", 0)
    return 0, 0, 0  # Default values if file doesn't exist
