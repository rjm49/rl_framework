'''
Created on 16 Oct 2016

@author: Russell
'''
from reinf import StudentTask

class FlashCard(StudentTask):
    '''
    This class represents an "adaptive flash-card"
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        self.grammar_level = 0
        self.vocab_level = 0
        
        