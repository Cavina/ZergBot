# Zerg AI with PySC2 and LLM Integration

## Overview
This project develops a **Zerg AI** using **DeepMind's PySC2** library. It integrates **reinforcement learning**, **decision-making translations via LLMs**, and **postgame analysis** to refine strategic gameplay in StarCraft II. It is under construction still. I am currently finishing up the reinforcement learning and planning for the next stages.

## Features
- **Reinforcement Learning (Q-learning-based)**: Implements AI logic for **Zerg race** decision-making.
- **Data Logging**: Captures game state transitions, actions, and rewards for later analysis.
- **LLM Integration**: Uses a **Large Language Model (LLM)** to translate game data into high-level strategic insights.
- **Postgame Analysis**: Processes logged data for performance evaluation and AI improvement.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- StarCraft II (SC2) client
- PySC2 (`pip install pysc2`)
- TensorFlow or PyTorch (for AI training)
- OpenAI API (for LLM processing, if applicable)

### Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure SC2 maps are available:
   ```bash
   python -m pysc2.bin.download_maps
   ```

## Usage
### Running the AI
Execute the Zerg AI from a terminal:
```bash
python run_smart_zerg_agent.py
```

### Logging Gameplay Data
Game events and state transitions are logged into a structured format (e.g., JSON, CSV). Logs can be processed post-game for insights.


## Roadmap
- [ ] Enhance AI decision-making using **deep Q-learning**.
- [ ] Improve **LLM translations** for better strategic explanations.
- [ ] Expand postgame analysis with **visualization tools**.
- [ ] Optimize performance for **real-time decision-making**.


## License
This project is licensed under the **MIT License**.

---
**Author:** Chris Avina
**GitHub:** [Cavina](https://github.com/Cavina)
