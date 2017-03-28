'''
Created on 7 Mar 2017

@author: Russell
'''
from asyncio.queues import LifoQueue

class HistoryModel(object):
    '''
    Basic state model
    '''

    def __init__(self, history_len, num_nodes):
        '''
        Constructor
        '''
        self.history = LifoQueue(maxsize=history_len) # history of previous States
        self.learned = [False] * num_nodes
 
    def _is_goal_state(self):
        return False not in self.learned
    
    def _update_history(self, c):
        self.history.put(c.id)
    
    def __str__(self):
        []+","+[i for i in self.history]
         