import random

import numpy
from collections import defaultdict


class Qutor():
    '''
    Tutoring agent with simple tabular state representation and Q-Learning type off-policy control algorithm
    '''

    def __init__(self, alpha=0.5, eps=10, gamma=0.95, actions=None, name="Qutor"):
        self.DEBUG = False
        # self.Q = {}  # here live the { S: [actions] } pairs for each tabular S...
        self.s_lookup = {}
        self.s_index = 0
        self.a_lookup = {}
        self.a_name_lookup={}
        self.actions = actions
        for aix, a in enumerate(self.actions):
            self.a_lookup[a] = aix
            self.a_name_lookup[aix] = a
        self.Qnp = numpy.ndarray(shape=(0, len(actions)))
        # self._Q = defaultdict(lambda: defaultdict(int))

        self.EPS = eps
        self.learn_rate = alpha
        self.gamma = gamma

        self.name = name
        self.transition_trace = []  # we keep the back history, of each move, of every episode, right here
        self.update_qvals = True
        self.prechosen = set()

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


    def initQS(self, s):
        if s not in self.s_lookup:
            #acquire an index for s
            # self.Qnp[self.s_index, :] = 0.0
            self.s_lookup[s] = self.s_index
            if self.Qnp.shape[0]<= self.s_index:
                #nouveau = numpy.random.rand(1000,self.Qnp.shape[1])
                nouveau = numpy.zeros((1000,self.Qnp.shape[1]))
                # nouveau = nouveau + 10.0
                print("stacked")
                self.Qnp = numpy.vstack((self.Qnp, nouveau))
            self.s_index += 1

        # if _s not in self.Q:
        #     self.Q[_s]={}
        #     if not(self.actions):
        #         print("Error, no actions in Qutor")
        #         exit(1)
        #     for a in self.actions:
        #         self.Q[_s][a] = 0.0 #random.random() # seed with random float in [0,1)


    def getQ(self, s,a):
        self.initQS(s)
        sx = self.s_lookup[s]
        ax = self.a_lookup[a]
        # if a not in self.Q[_s]:
        #     self.Q[_s][a] = 0
        return self.Qnp[sx, ax]

    def setQ(self, s,a, q):
        score = numpy.sqrt(numpy.dot(s,s))
        sx = self.s_lookup[s]
        ax = self.a_lookup[a]
        # if _s not in self.Q:
        #     self.Q[_s]={}
        # if a not in self.Q[_s]:
        #     self.Q[_s][a] = q
        was = self.Qnp[sx,ax]
        print(" -- - setting {}(sc{}),{} \t\t\t\t from {} : {}".format(sx,score,a,was,q))
        self.Qnp[sx,ax]=q

    def get_best_A_for_S(self, S):
        self.initQS(S)
        sx = self.s_lookup[tuple(S)]
        forS = self.Qnp[sx,:]
        # print(" ".join(map(str,S)),"="," ".join(map(str,forS)))
        # input("prmp")
        maxAxs = numpy.argwhere(forS == numpy.max(forS))
        # print(maxAxs)
        mAx = numpy.random.choice(maxAxs.flatten())
        return self.a_name_lookup[mAx]

    # def get_best_A_for_S(self, S):
    #     max_acts = None
    #     max_val = -float("inf")
    #     # print(self.Q[S])
    #     # print([a.id for a in actions])
    #     self.initQS(S)
    #     forS = self.Q[tuple(S)]
    #     # for a in forS:
    #     #     print(a,":",forS[a])
    #
    #     # maxA = max(forS, key=forS.get) # no key randomisation, not tiebreak :/
    #     max_QforS = max(forS.values())
    #     max_As = [k for (k, v) in forS.items() if v == max_QforS]
    #     maxA = random.choice(max_As)
    #     if self.DEBUG:
    #         print("max acts ", [a.id for a in max_acts])
    #         print("new rand of max acts=", maxA.id, "val=", max_val)
    #     # print("poss acts(", self.state_as_str(S),") = ", [(c.id, acts_dict[c]) for c in acts_dict])
    #     # max_key_by_val = max(acts_dict, key = acts_dict.get)
    #     #print("max for", S, "was",maxA,"with Q=",self.getQ(S,maxA))
    #     #input("promp")
    #     return maxA

    def sa_update(self, _S,A,R,nx_S):
        Q0 = self.getQ(_S,A)
        #print("old Q0=",Q0, "R is ",R)
        A1_best = self.get_best_A_for_S(nx_S)
        Q1_best = self.getQ(nx_S, A1_best)

        #print("!!",self.learn_rate, self.gamma, A1_best, Q1_best)
        #Q0 = (1-self.learn_rate)*Q0 + self.learn_rate*(R + self.gamma*Q1_best)
        Qdelta = self.learn_rate*(R + self.gamma*Q1_best - Q0)
        #print("Qdelta=",Qdelta)
        Q0 = Q0 + Qdelta
        #print("new Q0=",Q0)
        self.setQ(_S,A, Q0)

    # def sa_update(self, S, A, R, nx_S):
    #     Q0 = self.getQ(S,A)
    #     A1_best = self.get_best_A_for_S(nx_S)
    #     Q1_best = self.getQ(nx_S,A1_best)
    #     Q0 = (1-self.learn_rate)*Q0 + self.learn_rate*(R + self.gamma*Q1_best)
    #     # print("setting",S,A,"to", Q0)
    #     self.setQ(S,A, Q0)


    def status_report(self):
        sc=0
        ac=0
        sc = self.Qnp.shape[0]
        ac = self.Qnp.size

        print(sc, "states")
        print(ac, "s-a pairs")
        return sc, ac
