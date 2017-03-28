'''
Created on 13 Nov 2016

@author: Russell
'''
import sys
from reinf.exp1.students.base import BaseStudent

class IdealStudent(BaseStudent):
    '''
    This class simulates a human learner who is able to pick up new concept immediately and without forgetting,
    on the proviso that she already has the prerequisite skills to do so.
    
    The learner needs *all* the necessary prerequisites, it is not sufficient to have a subset.
    '''

    def try_learn(self, c):
        if self.can_learn(c):
            self.known_concepts[c]=1.0
            return True
        else:
            return False;
        
    def knows(self, c):
        if c in self.known_concepts:
            return self.known_concepts[c]
        else:
            return False;