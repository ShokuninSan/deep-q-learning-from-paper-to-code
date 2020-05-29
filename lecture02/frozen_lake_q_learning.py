import gym
import numpy as np
import matplotlib.pyplot as plt


N_EPISODES = 500_000
ALPHA = 0.001
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.01 


class Agent:

    def __init__(self, env):
        self.env = env 
        self.state = None
        self.Q = {
            (s, a): 0 
            for s in range(env.observation_space.n)
            for a in range(env.action_space.n)
        }

    def step(self, epsilon):
        action = None
        if np.random.rand() < epsilon:
            # do random action selection
            action = np.random.randint(env.action_space.n)
        else:
            # do greedy action selection
            action, _ = self.select_action_value(state) 

        new_state, reward, is_done, _ = env.step(action)

        self.update_action_value(state, action, reward, new_state)
        self.state = new_state

        return reward, is_done  

    def update_action_value(self, state, action, reward, new_state):
         _, value = self.select_action_value(new_state)
         self.Q[(state, action)] += ALPHA * (reward + GAMMA * value - self.Q[(state, action)])

    def select_action_value(self, state):
        max_action = None
        max_action_value = -1 
        for (s, a), value in self.Q.items():
            if s == state and value > max_action_value:
                max_action = a
                max_action_value = value
        return max_action, max_action_value


if __name__ == '__main__':

    rewards = []
    avg_100_rewards = []
    env = gym.make('FrozenLake-v0')
    agent = Agent(env)
    epsilon = EPSILON_START

    for episode in range(N_EPISODES):

        state = env.reset()
 
        while True:
 
            reward, is_done = agent.step(epsilon)

            if is_done:
                rewards.append(reward) 
                break
   
        epsilon = np.max([epsilon - (EPSILON_START/N_EPISODES), EPSILON_END]) 
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f'Average reward over last 100 episodes was {avg_reward}')
            avg_100_rewards.append(avg_reward)

    plt.plot(avg_100_rewards)
    plt.show()


