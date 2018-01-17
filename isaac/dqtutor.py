# Deep Q-learning Agent
from collections import deque
import random

from keras import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam
import numpy as np

from isaac import itemencoding


class DQTutor:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 1.0    # discount rate
        self.counter = 0
        self.explore_period = 5000
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.5
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model

        model = Sequential()
        # model.add(Dense(2*self.state_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(700, input_dim=self.state_size, activation='relu')) #, activation='relu'))
        model.add(Dense(700, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        exploratory = False
        if np.random.rand() <= self.epsilon:
            exploratory = True
            return random.randrange(self.action_size), exploratory
        state = np.array([state])
        act_values = self.model.predict(state)
        #print("act_values=",act_values)
        n_act = np.argmax(act_values[0])
        print("maxA,Q= ",n_act, np.max(act_values[0]))
        return np.argmax(act_values[0]), exploratory  # returns action

    def updateQ(self, state, action, reward, next_state, done, dec_eps=True):
        print("upd", state, action, reward, next_state)
        target = reward
        state = np.array([state])
        next_state = np.array([next_state])
        nxQ = self.model.predict(next_state)[0]
        amax = np.amax(nxQ)
        #print(nxQ, amax)
        if not done:
            target = reward + self.gamma * amax
        target_f = self.model.predict(state)
        target_f[0][action] = target
        #print("state fit:", state.shape, np.sum(state), target_f)
        #print(state)
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if dec_eps:
            if self.counter > self.explore_period:
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
            else:
                self.counter += 1
                print(self.counter)

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        # print("replay:")
        # print(minibatch)
        for state, action, reward, next_state, done in minibatch:
            self.updateQ(state, action, reward, next_state, done, dec_eps=False)
