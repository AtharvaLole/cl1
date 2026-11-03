
import gymnasium as gym
import random
import numpy as np

env=gym.make("Taxi-v4")
Q=np.zeros((env.observation_space.n,env.action_space.n))

for ep in range(10000):
    state,_=env.reset()
    for step in range(200):
        if random.random()<0.2:
            action=random.randint(0,5)
        else:
            action=np.argmax(Q[state])
        next_state,reward,done,truncated,_=env.step(action)
        Q[state,action]+=0.1*(
            reward+0.9*np.max(Q[next_state])-Q[state,action]
        )
        state=next_state
        if done or truncated:
            break

state,_=env.reset()
done=False
for step in range(200):
    action=np.argmax(Q[state])
    next_state,reward,done,truncated,_=env.step(action)
    state=next_state
    if done or truncated:
        break
print("ok" if done else "nope")
env.close()
