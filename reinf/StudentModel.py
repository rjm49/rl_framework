'''
Created on 16 Oct 2016

@author: Russell
'''

class StudentModel(object):
    '''
    This class models the human learner
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        self.grammar_level = 0
        self.vocab_level = 0