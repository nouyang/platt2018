import gym
import pybullet as p
import pybullet_envs
import numpy as np
import time

serverMode = p.GUI  # GUI/DIRECT
physicsClient = p.connect(serverMode)

env = gym.make('KukaCamBulletEnv-v0')
env.render('human')
env.reset()
# p.resetDebugVisualizerCamera(0.5, 0, -45, (0, 0, 0))


for i in range(10000):
    action = np.random.uniform(
        low=-1, high=1, size=(env.action_space.shape[0],))
    print(env.action_space.shape[0])
    axn = [0.8]*3
    action = env.step(np.array(axn))
    obs, reward, done, info = env.step(action)
    # time.sleep(0.2)
    # env.render('human')
