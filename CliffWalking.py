# OpenAI gym may show many errors so this is the latest solution
# pip install gym==0.25.2 tensorflow keras-rl2 numpy
# pip install pyglet
# pip install pygame
# pip install keras
# pip install protobuf==3.20.*
# use "from tensorflow.keras.optimizers.legacy import Adam"

import gym
import random
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from gym import spaces

env = gym.make("CliffWalking-v0", render_mode="human")

states = env.observation_space.n
actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)


agent.compile(Adam(lr=0.001), metrics=["mae"])
agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=10, visualize=False)
print(np.mean(results.history["episode_reward"]))

agent.save_weights('dqn_weights_cliff-walking.h5f')

env.close()

# episodes = 10
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0
#
#     while not done:
#         action = random.choice([0,1])       # 2 actions - LEFT/RIGHT
#         _, reward, done, _ = env.step(action)
#         score += reward
#         env.render()        # pyglet is necessary for this
#
#     print(f"Episode {episode}, Score: {score}")
#
# env.close()

