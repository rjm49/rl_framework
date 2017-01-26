'''
Created on 30 Nov 2016

@author: Russell
'''
from random import randint
import random


class TabularTutor(object):
    '''
    Tutoring agent with simple tabular state representation
    '''

    def __init__(self, num_nodes=100, alpha=0.5, eps=5):
        '''
        Constructor
        '''
        self.state = None 
        self.student_knowledge = [False for _ in range(num_nodes)]
        self.qs = {} #here live the { state: [actions] } pairs for each tabular state...
        self.EPS = eps
        self.learn_rate = alpha
        self.steps = 0
        
    def reset_steps(self):
        self.steps = 0
        
    def reset_knowledge(self):
        self.student_knowledge[:] = [False] * len(self.student_knowledge)

    
    def mission_complete(self):
        #print (self.student_knowledge)
        return (False not in self.student_knowledge)
    
    def student_learned(self, c, time_k):
        self.student_knowledge[c.id]=True
        
        rk = 100.0 if self.mission_complete() else 0.0 #check to see if we've won and our student is a Meisterin of her Kraft
        
        qs = self.qs
        state = self.state
        
        if (state in qs) and (c in qs[state]): #if this (state,action) has been experienced before, we update the value...
            qk           = qs[state][c]
            qk1          = qk + self.learn_rate*(rk - qk)/self.steps
            qs[state][c] = qk1
#            print("Updated est'd value {}->{} = {} from qk={} reward={} at step={}".format(state.id, c.id,qk1,qk,rk, self.steps))
#             print("Updated est'd value",state.id, "->", c.id, "=", qk1, "(",qk,rk, self.steps,")")
        else: # ...otherwise we introduce a new value
            if not state in qs:
                qs[state]={}
            qs[state][c]=rk
#             print("New estimated value {}->{} = {}".format(state.id if state else str(state), c.id, rk))
            
        self.state = c
        
    def pick_another(self, concepts):
        if(randint(1,self.EPS)==1 or self.state not in self.qs): #exploratory move
            return random.choice(concepts)
        else: #exploitative move
            poss_acts = self.qs[self.state]
            max_key_by_val = max(poss_acts, key = poss_acts.get)
            return max_key_by_val # return the most lucrative action for the current state


class RandomTutor(object):
    def __init__(self, num_nodes=100):
        self.student_knowledge = [False for _ in range(num_nodes)]
        self.steps = 0

    def reset_steps(self):
        self.steps = 0    

    def mission_complete(self):
        return (False not in self.student_knowledge)

    def student_learned(self, c_tried, time_k):
        self.student_knowledge[c_tried.id]=True
        
    def pick_another(self,concepts):
        return random.choice(concepts)