'''
Created on 2 Apr 2017

@author: Russell
'''

class MarkovianStateRep(object):
    '''
    classdocs
    '''

    def __init__(self, num_nodes):
        '''
        Constructor
        '''
        self.num_nodes = num_nodes
        self.len_hist = 3
#         self.S = None
#         self.lastS = None
    

    def reset_state(self):
        self.S = self.get_init_state()
        self.lastS = None
        return self.S
        
    def get_init_state(self):
        return [None]*self.len_hist
    
    def get_next_state(self, S,A, succ):
        cp = S[1:]
        cp.append(A.id)
        return cp
    
    def _check_goal_state(self, new_S):
        pass
    
    def __str__(self):
        s = ",".join([str(a) for a in self.S])
        return s