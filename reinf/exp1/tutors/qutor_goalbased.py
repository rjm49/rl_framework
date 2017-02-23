'''
Created on 30 Nov 2016

@author: Russell
'''
from reinf.exp1.policies.policy_utils import state_as_str
from reinf.exp1.domains.filterlist_utils import update_filter
from reinf.exp1.tutors.base import BaseTutor


class QutorGoalTutor(BaseTutor):
    '''
    Tutoring agent with simple tabular state representation and SARSA type on-policy control algorithm
    '''


    def __init__(self, num_nodes=100, alpha=0.5, eps=0.01, gamma=1.0, name="SarsaTutor"):
        '''
        Constructor
        '''
        super().__init__()
        self.DEBUG=False
        #self.student_knowledge = [False for _ in range(num_nodes)]
        self.Q = {} #here live the { thisS: [actions] } pairs for each tabular thisS...
        self.EPS = eps
        self.learn_rate = alpha
        self.gamma = gamma
        self.num_nodes = num_nodes
                
        self.thisS = tuple([False] * num_nodes)
        self.lastA = None
        self.lastR = None
        
        self.filterlist = {}
        self.name = name


    def sa_update(self, S,A,R,nx_S, concepts):
        Q0 = self.Q[S][A]
        A1_best = self.get_best_A_for_S(nx_S, concepts)
        Q1_best = self.Q[nx_S][A1_best]
        Q0 = Q0 + self.learn_rate*(R + self.gamma*Q1_best - Q0)
        self.Q[S][A] = Q0

    def run_episode(self, model, stu, max_steps=-1, update_qvals=True):
        actions = model.concepts
#         lastS = None
        S = tuple([False] * len(actions)) #reset the tutor's state .. we assume the student knows nothing
        self.extend_Q(S, actions)
        
        self._new_trace()

        max_steps = float('inf') if max_steps<=0 else max_steps # hack up an infinite number of attempts here
        step_cnt=0
        
        while (max_steps<=0 or step_cnt<=max_steps) and (False in S):
            A,exp = self.choose_A(S, actions)
            
            if A.id not in self.filterlist:
                self.filterlist[A.id]= [True]*len(actions)
    
            succ = stu.try_learn(A)
            if succ:
                update_filter(self.filterlist, S, A.id, succ)
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
            
            #lastS = S
            S = new_S
            step_cnt+=1
            print(state_as_str(self.thisS), step_cnt)
        print("Qutor: Episode over in",step_cnt,"steps")
        return step_cnt