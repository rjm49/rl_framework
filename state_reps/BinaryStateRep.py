'''
Created on 2 Apr 2017

@author: Russell
'''

class BinaryStateRep(object):
    '''
    classdocs
    '''

    def __init__(self, num_nodes):
        '''
        Constructor
        '''
        self.num_nodes = num_nodes
        self.reset_state()
    
    def reset_state(self):
        self.S = self.get_init_state()
        return self.S

    def get_init_state(self):
        return [False] * self.num_nodes
    
    def get_next_state(self, S,A, succ):
        cp = list(S)
        cp[A.id]=succ
        return cp
        
    def _check_goal_state(self, new_S):
        return (False not in new_S)
    
    def __str__(self):
        s = "".join([("1" if x else "0") for x in self.S ])
        return s