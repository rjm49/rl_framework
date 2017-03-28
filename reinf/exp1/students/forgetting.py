'''
Created on 13 Nov 2016

@author: Russell
'''
import sys
import random
from _collections import defaultdict
from enum import Enum
from math import log
from reinf.exp1.students.base import BaseStudent

class ForgetMode(Enum):
    EXPONENTIAL = 0
    EBBINGHAUS_POWER = 1
    EBBINGHAUS_LOG = 2
    WICKELGREN_POWER = 3
    KRUEGER_POWER = 4 # Based on Krueger 1929

class ForgettingStudent(BaseStudent):
    '''
    This class simulates a student who forgets concepts with decay F .. this governs probability of recall
    '''
    def __init__(self):
        super().__init__()
        self.forget_mode = ForgetMode.EBBINGHAUS_LOG
        self.TICK_SIZE = 1 # how many minutes does a system tick represent
        self.MASTERY_THRESHOLD = 0.9
        if self.forget_mode==ForgetMode.EBBINGHAUS_LOG:
            self.ret_fn = self._get_retention_ebb
        else:
            self.ret_fn = self._get_retention_exp        

    def _decay_tick(self, tsize):        
        for c in self.known_concepts:
            t = self.time_since_learnt[c] + tsize
            k_c = self.ret_fn(t)
            self.time_since_learnt[c] = t
            self.known_concepts[c] = k_c
#             print(c.id,"=",k_c, " time since learned = " , t)

    def try_learn(self, c):
        self._decay_tick(self.TICK_SIZE)
        if self.can_learn(c):
#             if c in self.known_concepts:
#                 print("----> refresher: ",c.id,"@", self.known_concepts[c], self.time_since_learnt[c])
#             else:
#                 print("***new",c.id)
            self.known_concepts[c]=1.0
            self.time_since_learnt[c]=0.0
            return True
        else:
#             print("fail",c.id)
            return False

    def knows(self, c):        
        if c in self.known_concepts:
            p_knowing = self.known_concepts[c]
            r = random.random()
            if r < p_knowing:
#                 print(r, "<", p_knowing)
                return True;
        return False    

    def _get_retention_ebb(self, t):
        if t<=1:
            return 1.0
        else:
            c = 0.74617452
            d = 0.13578415
            innr = pow( log(t, 10.0), d)
            return c / (innr + c)  

    def _get_retention_exp(self, t):
        if t==0:
            return 1.0
        else:
            return pow( 0.999, t )
            

#     def get_knowledge(self):
#         k = [self.known_concepts[c] for c in self.known_concepts]
#         return k