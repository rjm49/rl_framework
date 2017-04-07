'''
Created on 3 Mar 2017

@author: Russell
'''
from reinf.exp1.tutors.qutor import Qutor
from _collections import defaultdict
from reinf.exp1.policies.policy_utils import state_as_str
import random
from profile_decr import profile
from state_reps.BinaryStateRep import BinaryStateRep
from state_reps.MarkovianStateRep import MarkovianStateRep

class DynaQutor2(Qutor):
    '''
    classdocs
    '''

    def __init__(self, num_nodes=100, alpha=0.5, eps=100, gamma=1.0, name="DynaQutor"):
        '''
        Constructor
        '''
        super().__init__(num_nodes, alpha, eps, gamma, name)
        self.model = defaultdict(lambda: defaultdict(lambda: None))
        self.MASTERY_THRESHOLD=0.95
        self.sRep = BinaryStateRep(num_nodes)
#         self.sRep = MarkovianStateRep(num_nodes)
        self.modelling_intensity = 10

    def test_student(self, p):
        S = self.sRep.S
        for A in self.possible_actions:
            suc = p.knows(A)
            S[A.id]=suc
        self.extend_Q(S, self.possible_actions)
    
    def _update_model(self, s, a, rew, S_nx):
        s= tuple(s)
        self.model[s][a] = (rew, S_nx)
    
    def _query_model(self, s, a):
        RS_pair = self.model[s][a]
        return RS_pair

    def reset(self):
        self.sRep.reset_state()

#     def _check_goal_state(self, new_S):
#         return (False not in new_S)
        
    def record_lesson_results(self, lesson, succ, was_known, mastery, was_exploratory):
        S=self.sRep.S
        A=lesson
        
        new_S = self.sRep.get_next_state(S,A, succ)
        self.extend_Q(new_S, self.possible_actions)
#         self._add_to_trace(S, A, succ)
        
        R = -1.0
        
#         if succ:
#             if not was_known:
#                 R = 1.0
#             else:
#                 R = -1.0
#         else:
#             R = -2.0
            
#         if self._check_goal_state(new_S):
        if mastery>=self.MASTERY_THRESHOLD:
            R=10000.0
#         else:
#             new_S = S

        if self.update_qvals:
            self.sa_update(S, A, R, new_S, self.possible_actions)
            if self.modelling_intensity > 0:
                self._update_model(S, A, R, new_S)        
                model_keys = list(self.model.keys())
                for _ in range(self.modelling_intensity):
                    rand_S = random.choice(model_keys)
                    rand_A = random.choice(list(self.model[rand_S].keys()))
                    pred_R, pred_nx_S  = self._query_model(rand_S, rand_A)
                    self.sa_update(rand_S, rand_A, pred_R, pred_nx_S, self.possible_actions)

        self.sRep.S = new_S
    