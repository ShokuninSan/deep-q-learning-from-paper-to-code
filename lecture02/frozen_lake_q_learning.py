import gym
import numpy as np
import matplotlib.pyplot as plt


N_GAMES = 500_000
ALPHA = 0.001
GAMMA = 0.9
EPSILON_START = 0.01
EPSILON_END = 1.0


class Agent:

    def __init__(self):
        self.state = None
        self.Q = {
            (s, a): 0 
            for s in range(self.env.observation_space.n)
            for a in range(self.env.action_space.n)
        }

    def step(self, state, env, epsilon):
        action = None
        if np.random.rand() > epsilon:
            # do random action selection
            action = np.random.randint(env.action_space.n)
        else:
            # do greedy action selection
            action, _ = self._search_max_Q(state) 

        new_state, reward, is_done, _ = env.step(action)
       
        _, max_Q_state_prime = self._search_max_Q(new_state)

        Q = self.Q[(self.state, action)]

        self.Q[(self.state, action)] = Q + ALPHA * (reward + GAMMA * max_Q_state_prime - Q) 

        self.state = new_state

    def _search_max_Q(self, state):
        max_action = None
        max_action_value = -1 
        for (s, a), value in self.Q.items():
            if s == state and value > max_action_value:
                max_action = a
                max_action_value = value
        return max_action, max_action_value

scores = []

for g in range(N_GAMES):
    
    while True:
        
