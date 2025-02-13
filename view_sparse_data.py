import pandas as pd
import zerg_actions

DATA_FILE = 'sparse_agent_data.gz'

q_table = pd.read_pickle(DATA_FILE, compression='gzip')

best_action_indices = q_table.idxmax(axis=1)  # Get column with the highest Q-value

print(q_table[0][0].dtypes)
# q_table.to_csv("q_table.csv", index=True)
# print("Q-table saved as q_table.csv")