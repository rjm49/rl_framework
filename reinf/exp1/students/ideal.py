'''
Created on 13 Nov 2016

@author: Russell
'''
import sys

class IdealLearner(object):
    '''
    This class simulates a human learner who is able to pick up new concept immediately and without forgetting,
    on the proviso that she already has the prerequisite skills to do so.
    
    The learner needs *all* the necessary prerequisites, it is not sufficient to have a subset.
    '''


    def __init__(self):
        '''
        We are born in a state of blissful ignorance
        '''
        self.known_concepts = []
        
    def try_learn(self, c):
        if self.can_learn(c):
            self.known_concepts.append(c)
            return True
        else:
            return False
    
    def reset_knowledge(self):
        self.known_concepts = []
    
    def can_learn(self, c):
        if self.knows(c): # already known
            sys.stdout.write(str(c.id)+".AK ")
            return False
        for p in c.predecessors:
            if not self.knows(p): # do not have all the requisite background knowledge
                sys.stdout.write(str(c.id)+".PU."+str(p.id)+" ")
                return False        
        return True #good to learn
    
    def knows(self, c):
        return True if (c==None or c in self.known_concepts) else False
            