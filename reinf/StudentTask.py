'''
Created on 16 Oct 2016

@author: Russell
'''

class StudentTask(object):
    '''
    This represents a generic "learning task" for the human learner (student), as opposed to a machine learner (agent)
    '''

    def __init__(self, params):
        '''
        Constructor
        '''
        self.proficiency_est = 0
        self.relevances = []

    def assess(self):
        score = 0
        return score