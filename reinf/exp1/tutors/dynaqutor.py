'''
Created on 3 Mar 2017

@author: Russell
'''
from reinf.exp1.tutors.qutor import Qutor
from _collections import defaultdict
from reinf.exp1.policies.policy_utils import state_as_str
import random

class DynaQutor(Qutor):
    '''
    classdocs
    '''


    def __init__(self, num_nodes=100, alpha=0.5, eps=100, gamma=1.0, name="DynaQutor"):
        '''
        Constructor
        '''
        super().__init__(num_nodes, alpha, eps, gamma, name)
        self.model = defaultdict(lambda: defaultdict(lambda: None))
    
    def _update_model(self, s, a, rew, S_nx):
        self.model[s][a] = (rew, S_nx)
    
    def _query_model(self, s, a):
        RS_pair = self.model[s][a]
        return RS_pair
    
    def run_episode(self, model, stu, max_steps=-1, update_qvals=True):
        actions = model.concepts

        S = tuple([False] * len(actions)) #reset the tutor's state .. we assume the student knows nothing
        self.extend_Q(S, actions)
        
        self._new_trace()

        max_steps = float('inf') if max_steps<=0 else max_steps # hack up an infinite number of attempts here
        step_cnt=0
        
        while (max_steps<=0 or step_cnt<=max_steps) and (False in S):
            A,explor = self.choose_A(S, actions)
            
#             if A.id not in self.filterlist:
#                 self.filterlist[A.id]= [True]*len(actions)
    
            succ = stu.try_learn(A)
            if succ:
#                 update_filter(self.filterlist, S, A.id, succ)
                R=-1.0
                new_S = self.get_next_state(S,A)
                self._add_to_trace(S, A, True)
                self.extend_Q(new_S, actions)
                if (False not in new_S):
                    R=10000.0
            else:
                R=-1.0
                new_S = S
                self._add_to_trace(S, A, False)
            
            if update_qvals:
                self.sa_update(S, A, R, new_S, actions)
                self._update_model(S, A, R, new_S)
            
            model_keys = list(self.model.keys())
            for _ in range(10):
                rand_S = random.choice(model_keys)
                rand_A = random.choice(list(self.model[rand_S].keys()))
#                 print("rand S A", rand_S, rand_A)
                #rs_pair = self._query_model(rand_S, rand_A)
                pred_R, pred_nx_S  = self._query_model(rand_S, rand_A)
                #if rs_pair:
                #    pred_R, pred_nx_S = rs_pair
#                     print("pred RS=", rs_pair)
                self.sa_update(rand_S, rand_A, pred_R, pred_nx_S, actions)
                
            S = new_S
            step_cnt+=1
            #print(state_as_str(S), step_cnt)
        print("DynaQutor: Episode over in",step_cnt,"steps")
        return step_cnt