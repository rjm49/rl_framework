# Deep Q-learning Agent
from collections import deque
import random

from keras import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.optimizers import Adam, RMSprop
import numpy as np
from keras.utils import plot_model

from isaac import itemencoding
override_input_dim = 202

class DQTutor:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 1.0    # discount rate
        self.counter = 0
        self.explore_period = 5000
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.1
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(500, input_dim=override_input_dim)) #, activation='relu')) #300,100
        model.add(Dense(500))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam())
        plot_model(model, to_file='model.png')
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        exploratory = False
        if np.random.rand() <= self.epsilon:
            exploratory = True
            return random.randrange(self.action_size), exploratory
        state = np.array([state[1:override_input_dim+1]])
        act_values = self.model.predict(state)
        #print("act_values=",act_values)
        n_act = np.argmax(act_values[0])
        print("maxA,Q= ",n_act, np.max(act_values[0]))
        return np.argmax(act_values[0]), exploratory  # returns action

    def updateQ(self, state, action, reward, next_state, done, dec_eps=True):
        # print("upd", "".join(map(str,map(int,10*state))))
        print(action, reward)
        #print("upd",state, action, reward,next_state)
        state = np.array([state[1:override_input_dim+1]])
        if done:
            target = reward
        else:
            next_state = np.array([next_state[1:override_input_dim+1]])
            nxQ = self.model.predict(next_state)[0]
            amax = np.amax(nxQ)
            # print(nxQ, amax)
            print("AMAX:", amax)
            target = reward + self.gamma * amax
        target_f = self.model.predict(state)
        target_f[0][action] = target
        # print("fitting {} with {}".format(state, target_f))
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if(dec_eps):
            if(self.counter > self.explore_period):
                self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        # print("replay:")
        # print(minibatch)
        for state, action, reward, next_state, done in minibatch:
            self.updateQ(state, action, reward, next_state, done, dec_eps=False)
