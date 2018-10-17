import gym
import numpy as np
from dqnPoleAgent import DQNAgent

env = gym.make("CartPole-v1")
env.reset()


env._max_episode_steps = 1000

agent = DQNAgent(input_dim=4, output_dim=2)

state, reward, done, _ = env.step(env.action_space.sample())


# Play the game many times
for ep in range(0, 500): # 500 episodes of learning
    total_reward = 0 # Maintains the score for this episode.

    while True:
        env.render() # Show the animation of the cartpole
        action = agent.act(state) # Get action
        next_state, reward, done, _ = env.step(action) # Take action
        total_reward += reward # Accrue reward

        if done: # Episode is completed due to failure or cap being reached.
            print("Episode: {}, Total reward: {}, Explore P: {}".format(ep, total_reward, agent.explore_p))
            if total_reward == 999: # Simulation completed without failure. Save a copy of this network.
                agent.model.save("cartpole.h5")
            # Add experience to bucket (next_state is None since epoch is over).
            agent.remember(state, action, None, reward)
            env.reset() # Reset environment
            break

        else: # Episode not over.
            agent.remember(state, action, next_state, reward) # Store tuple.
            state = next_state # Advance state
            agent.replay() # Train the network form replay samples.

