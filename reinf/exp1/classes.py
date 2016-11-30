'''
Created on 11 Nov 2016

@author: Russell
'''
class Concept:
    '''
    classdocs
    '''
    counter = 0
    
    def __init__(self, _id):
        '''
        Constructor
        '''
        self.predecessors = []
        self.id = _id
#         self.id = Concept.counter
#         Concept.counter+=1

    def has_ancestor(self, poss_anc): #note only checks historical connections (i.e. cycles in graph) not sameness of self==concept
        
#         print(poss_anc.id,"->*-",self.id, [p.id for p in self.predecessors], "?")

        if not self.predecessors: 
            return False
        
        if(poss_anc in self.predecessors):
            return True
        #see if "concept" is in the family tree of any of our predecessors
        
        for p in self.predecessors:
            if p.has_ancestor(poss_anc):
                return True
        #not in direct preds, not in families, so return false...

        return False