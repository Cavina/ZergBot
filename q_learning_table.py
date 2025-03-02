import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


    def choose_action(self, observation):
        self.check_state_exists(observation)

        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]

            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action
    

    def learn(self, s, a, r, s_):
        self.check_state_exists(s_)
        self.check_state_exists(s)

        q_predict = self.q_table.loc[s, a]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r

        self.q_table.loc[s, a] += self.lr * (q_target-q_predict)

    def check_state_exists(self, state):
        if state not in self.q_table.index:
            new_row = pd.DataFrame([[0] * len(self.actions)], columns=self.q_table.columns, index=[state])
            self.q_table = pd.concat([self.q_table, new_row])
