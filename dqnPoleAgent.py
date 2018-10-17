# https://quantsoftware.gatech.edu/CartPole_DQN

import numpy as np
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.models import Model
from collections import deque


class DQNAgent:

    def __init__(self, input_dim, output_dim, learning_rate=.005,
        mem_size=5000, batch_size=64, gamma=.99, decay_rate=.0002):

    # Save instance variables.
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.batch_size = batch_size
    self.gamma = gamma
    self.decay_rate = decay_rate

    # Define other instance variables.
    self.explore_p = 1 # The current probability of taking a random action.
    self.memory = deque(maxlen=mem_size) # Define our experience replay bucket as a deque with size mem_size.

    # Define and compile our DQN. This network has 3 layers of 24 nodes. This is sufficient to solve
    # CartPole, but you should definitely tweak the architecture for other implementations.
    input_layer = Input(shape=(input_dim,))
    hl = Dense(24, activation="relu")(input_layer)
    hl = Dense(24, activation="relu")(hl)
    hl = Dense(24, activation="relu")(hl)
    output_layer = Dense(output_dim, activation="linear")(hl)
    self.model = Model(input_layer, output_layer)
    self.model.compile(loss="mse", optimizer=RMSprop(lr=learning_rate))

    def act(self, state):
    # First, decay our explore probability
    self.explore_p *= 1 - self.decay_rate
    # With probability explore_p, randomly pick an action
    if self.explore_p > np.random.rand():
        return np.random.randint(self.output_dim)
    # Otherwise, find the action that should maximize future rewards according to our current Q-function policy.
    else:
        return np.argmax(self.model.predict(np.array([state]))[0])

def remember(self, state, action, next_state, reward):
    # Create a blank state. Serves as next_state if this was the last experience tuple before the epoch ended.
    terminal_state = np.array([None]*self.input_dim) 
    # Add experience tuple to bucket. Bucket is a deque, so older tuple falls out on overflow.
    self.memory.append((state, action, terminal_state if next_state is None else next_state, reward))


def replay(self):

    # Only conduct a replay if we have enough experience to sample from.
    if len(self.memory) < self.batch_size:
        return

    # Pick random indices from the bucket without replacement. batch_size determines number of samples.
    idx = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
    minibatch = np.array(self.memory)[idx]
    self.train(minibatch)
    
    # Extract the columns from our sample
    states = np.array(list(minibatch[:,0]))
    actions = minibatch[:,1]
    next_states = np.array(list(minibatch[:,2]))
    rewards = np.array(minibatch[:,3])

    # Compute a new estimate for each Q-value. This uses the second half of Bellman's equation.
    estimate = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)

    # Get the network's current Q-value predictions for the states in this sample.
    predictions = self.model.predict(states)
    # Update the network's predictions with the new predictions we have.
    for i in range(len(predictions)):
        # Flag states as terminal (the last state before a epoch ended).
        terminal_state = (next_states[i] == np.array([None]*self.input_dim)).all()
        # Update each state's Q-value prediction with our new estimate.
        # Terminal states have no future, so set their Q-value to their immediate reward.
        predictions[i][actions[i]] = rewards[i] if terminal_state else estimate[i]

    # Propagate the new predictions through our network.
    self.model.fit(states, predictions, epochs=1, verbose=0)

