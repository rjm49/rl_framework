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
from reinf.exp1.students.forgetting import ForgettingStudent, ForgetMode

class RelearningStudent(ForgettingStudent):
    '''
    This class simulates a student who forgets concepts with decay F .. this governs probability of recall
    '''
    def __init__(self):
        super().__init__()
        self.relearn_times = {} # type: dict<list of int times>
        self.relearn_weights = {}
        self.t = 0
        
    def _decay_tick(self, tsize):        
        self.t += tsize
        for c in self.known_concepts:
            k_c = self._get_composite_retention(c, self.t)
            self.known_concepts[c] = k_c
#             print(c.id,"=",k_c, " time since learned = " , t)

    def _get_composite_retention(self, c,t):
        rf = self.ret_fn
        sum_r = 0.0
        for rlt,rlw in zip(self.relearn_times[c], self.relearn_weights[c]):
            ret = rf(t - rlt)
#             print("ret @ {}-{} -> {}".format(t,rlt,ret))
            sum_r += rlw*ret
#         print("cr for {} @ {} = {}".format(c.id,t,sum_r))
        return sum_r

    def try_learn(self, c):
        self._decay_tick(self.TICK_SIZE)
        if self.can_learn(c):
#             if c in self.known_concepts:
#                 print("----> refresher: ",c.id,"@", self.known_concepts[c], self.time_since_learnt[c])
#             else:
#                 print("***new",c.id)
           
            t = self.t
            if c not in self.known_concepts:
                self.known_concepts[c]=1.0
                self.relearn_times[c]=[t]
                self.relearn_weights[c]=[1.0]
            else:
                k_c = self._get_composite_retention(c, t)
                new_w = (1.0 - k_c) # sets the proportion weight for this new concept
                self.relearn_times[c].append(t)
                self.relearn_weights[c].append(new_w)    
                self.known_concepts[c] = 1.0
#                 print("relearned {} @ {} with new_w {} ... ret was {}".format(c.id, t, new_w, k_c))
            return True
        else:
#             print("fail",c.id)
            return False


    
#     def get_knowledge(self):
#         k = [self.known_concepts[c] for c in self.known_concepts]
#         return k