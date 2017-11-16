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
from reinf.exp1.tutors.base import BaseTutor

class SarsaTutor2(BaseTutor):
    '''
    classdocs
    '''

    def __init__(self, num_nodes=100, alpha=0.5, eps=100, gamma=1.0, name="SarsaTutor2"):
        '''
        Constructor
        '''
        super().__init__(num_nodes, alpha, eps, gamma, name)
        self.model = defaultdict(lambda: defaultdict(lambda: None))
        self.MASTERY_THRESHOLD=0.95
        self.sRep = BinaryStateRep(num_nodes)
#         self.sRep = MarkovianStateRep(num_nodes)
        self.modelling_intensity = 10

        self.history = []
        self.lambda_val = 0.9
        self.history_limit = 1
        
        gxl=self.gamma*self.lambda_val
        if self.lambda_val==0.0:
            self.history_limit=1
        elif gxl<1.0:
            i = 0
            while(gxl >= 0.0001):
                print(gxl)
                gxl *= self.gamma * self.lambda_val
                i+=1
            self.history_limit = i
        else:
            self.history_limit = 100
        print("lambda = ", self.lambda_val)
        print("histoty limit is",self.history_limit)


    def get_next_lesson(self):
        if self.A:
            return self.A
        else:
            self.choose_A(self.sRep.S, self.possible_actions)

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
    
    def sarsa_update(self, S,A,R,nx_S,nx_A):
        Q = self.Q[S][A]
        nx_Q = self.Q[nx_S][nx_A]
        delta = R + self.gamma*nx_Q - Q
        learnage = self.learn_rate * delta
        z_scaling = 1.0
        seen=set()
        for sa in self.history[:self.history_limit]:
            if tuple(sa) not in seen:
                s = sa[0]
                a = sa[1]
                self.Q[s][a] += learnage*z_scaling
                seen.add(tuple(sa))
            z_scaling = z_scaling * (self.gamma * self.lambda_val) # g*l==decay

    
    def record_lesson_results(self, lesson, succ, was_known, mastery, was_exploratory):
        S=self.sRep.S
        A=lesson
        
        new_S = self.sRep.get_next_state(S,A, succ)
        self.extend_Q(new_S, self.possible_actions)
        new_A = self.choose_A(new_S, self.possible_actions)
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
            self.sarsa_update(S, A, R, new_S, new_A)
#             if self.modelling_intensity > 0:
#                 self._update_model(S, A, R, new_S)        
#                 model_keys = list(self.model.keys())
#                 for _ in range(self.modelling_intensity):
#                     rand_S = random.choice(model_keys)
#                     rand_A = random.choice(list(self.model[rand_S].keys()))
#                     pred_R, pred_nx_S  = self._query_model(rand_S, rand_A)
#                     self.sarsa_update(rand_S, rand_A, pred_R, pred_nx_S, self.possible_actions)

        self.sRep.S = new_S
        self.A = new_A