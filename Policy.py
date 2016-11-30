'''
Created on 21 Oct 2016

@author: Russell
'''

class Policy(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        self.act_vals = {}
        self.state_vals = {}
        
    def get_valid_states(self):
        ''' the states that are recognised by this policy '''
        return self.state_vals.keys()
    
    def get_valid_actions(self, state):
        return self.state_vals
        
    def val(self, state):
        ''' return the (estimated?) value of a state, as evaluated under this policy '''
        return self.state_vals[state]
    
    def act_val(self, state, action):
        ''' return the (estimated?) value of a state-action pair as evaluated under this policy '''
        return self.act_vals[(state, action)]