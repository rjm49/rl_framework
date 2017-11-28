import random

import numpy
from collections import defaultdict

from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPRegressor

from isaac.itemencoding import gen_qenc


class QutorFA():
    '''
    Tutoring agent with simple tabular state representation and Q-Learning type off-policy control algorithm
    '''

    def __init__(self, alpha=0.5, eps=10, gamma=0.95, actions=None, name="QutorFA", cat_lookup=None, cat_ixs=None, passrates=None, stretches=None, levels=None, qpassquals=None):
        self.DEBUG = False
        # self.Q = {}  # here live the { S: [actions] } pairs for each tabular S...
        self.actions = actions
        self.Q = MLPRegressor(hidden_layer_sizes=(10,))

        self.EPS = eps
        self.learn_rate = alpha
        self.gamma = gamma

        self.name = name
        self.transition_trace = []  # we keep the back history, of each move, of every episode, right here
        self.update_qvals = True

        self.cat_lookup = cat_lookup
        self.cat_ixs = cat_ixs
        self.passrates = passrates
        self.stretches = stretches
        self.levels = levels
        self.qpassquals = qpassquals

    def choose_A(self, _S):
        if self.EPS>0:
            explore = random.randint(0, self.EPS) == 0
        else:
            explore = False
        if (explore): # starting move OR exploratory move
            a = random.choice(self.actions)
            print("explore",a)
        else: #exploitative move
            a = self.get_best_A_for_S(_S)
            print("exploit",a)
        return a, explore

    def getQ(self, s,a):
        a_qenc = self._get_qenc(a)
        inp = numpy.append(s.flatten(), a_qenc.flatten())
        Q_ = -1.0
        try:
            Q_ = self.Q.predict(inp.reshape(1,-1))
        except NotFittedError as e:
            #print("Q not initialised")
            Q_ = 0.0
        print("Q=",Q_)
        return Q_

    def setQ(self, s,a, q):
        _a = self._get_qenc(a)
        print(s)
        print(s.shape)
        print(_a)
        print(_a.shape)
        enc = numpy.append(s.flatten(), _a.flatten()).reshape(1,-1)
        print(enc.shape)
        print("Set Q", q)
        self.Q.partial_fit(enc, [q])
        print("new Q is", self.Q.predict(enc))

    def _get_qenc(self, A):
        cx = self.cat_ixs[self.cat_lookup[A]]
        pr = self.passrates[A]
        st = self.stretches[A]
        lv = self.levels[A]
        ql = self.qpassquals[A]
        return gen_qenc(cx, pr, st, lv, ql)

    def get_best_A_for_S(self, S):
        max_Q = float("-inf")
        max_A = random.choice(self.actions)
        for A in self.actions:
            Q = self.getQ(S,A)
            if max_Q < Q:
                max_Q = Q
                max_A = A
            return max_A

    def sa_update(self, S,A,R,nx_S):
        Q0 = self.getQ(S,A)
        #print("old Q0=",Q0, "R is ",R)
        A1_best = self.get_best_A_for_S(nx_S)
        Q1_best = self.getQ(nx_S, A1_best)

        #print("!!",self.learn_rate, self.gamma, A1_best, Q1_best)
        #Q0 = (1-self.learn_rate)*Q0 + self.learn_rate*(R + self.gamma*Q1_best)
        Qdelta = self.learn_rate*(R + self.gamma*Q1_best - Q0)
        #print("Qdelta=",Qdelta)
        Q0 = Q0 + Qdelta
        #print("new Q0=",Q0)
        self.setQ(S,A, Q0)

    def status_report(self):
        return 0, 0