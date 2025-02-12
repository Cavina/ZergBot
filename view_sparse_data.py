import pandas as pd

DATA_FILE = 'sparse_agent_data.gz'

q_table = pd.read_pickle(DATA_FILE, compression='gzip')

q_table.to_csv("q_table.csv", index=True)
print("Q-table saved as q_table.csv")