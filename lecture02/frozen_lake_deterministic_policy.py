import gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

n_games = 1000

policy = {0: 1, 1:0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 2, 9: 1, 10: 1, 11: 0, 12: 0, 13: 2, 14: 2, 15: 0}

win_pct = [] 
rewards = [] 
for g in range(n_games): 
    state = env.reset() 
    while True: 
        new_state, reward, is_done, _ = env.step(policy[state]) 
        state = new_state 
        if is_done: 
            rewards.append(reward) 
            break 
        if g % 10 == 0: 
            win_pct.append(sum(rewards[-10:]) / 10) 

plt.plot(win_pct)
plt.show()
