'''
Created on 30 Jan 2017

@author: Russell
'''
class AbstractTutor(object):
    def __init__(self, num_nodes=100, name=""):
        self.name=name
        pass

    def reset_steps(self):
        pass

    def mission_complete(self):
        pass

    def student_tried(self, c_tried):
        pass
     
    def choose_A(self,concepts):
        pass

    def get_policy(self):
        pass