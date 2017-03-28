'''
Created on 22 Mar 2017

@author: Russell
'''

import sys
from _collections import defaultdict

class BaseStudent(object):

    def __init__(self):
        '''
        We are born in a state of blissful ignorance
        '''
        self.known_concepts = {}
        self.time_since_learnt = defaultdict(int)
        self.ALLOW_RELEARNING = True
        
    def try_learn(self, c):
        pass
    
    def reset_knowledge(self):
        self.known_concepts.clear()
    
    
    def can_learn(self, c):
        if self.knows(c): # already known#
#             if self.ALLOW_RELEARNING:
                return True
#             else:
#                 return False        
        for p in c.predecessors:
#             print(c.id, " needs ", p.id)
            if not self.knows(p): # do not have all the requisite background knowledge
                return False        
        return True #good to learn
    
    def knows(self, c):
        pass
    
    def get_mastery_score(self):
        K = [self.known_concepts[c] for c in self.known_concepts]
#         print(K)
        return sum(K)
        