'''
Created on 30 Nov 2016

@author: Russell
'''
from random import randint
import random
from reinf.exp1.policies.policy_utils import state_as_str
from abc import abstractmethod
from _collections import defaultdict
from state_reps.BinaryStateRep import BinaryStateRep

class BaseStateRep():
    def __init__(self):
        self.S = None
    def __str__(self):
        return "BaseStateRep has no state... must be subclassed"

class BaseTutor(object):

    def __init__(self, num_nodes=100, alpha=0.5, eps=100, gamma=1.0, name="BaseTutor"):
        '''
        Constructor
        '''
        self.DEBUG=False
        self.Q = {} #here live the { S: [actions] } pairs for each tabular S...
        #self._Q = defaultdict(lambda: defaultdict(int))
        self.EPS = eps
        self.learn_rate = alpha
        self.gamma = gamma
        self.num_nodes = num_nodes
                

        self.name = name
        self.transition_trace = [] #we keep the back history, of each move, of every episode, right here
    
        self.student = None
        self.update_qvals = True
        self.sRep = BaseStateRep()
    
    def __str__(self):
        return "{} e{} a{} g{}".format(self.name,self.EPS,self.learn_rate,self.gamma)
    
#     def reset(self):
#         self.S = tuple([False] * self.num_nodes)
#         self.lastS = ()
#       
    #BEGIN EPISODE TRACING CODE    
    def _new_trace(self):
        self.transition_trace.append([])
    def _add_to_trace(self, S, A, passed=True ):
        self.transition_trace[-1].append(tuple([state_as_str(S),A.id, passed]))
    #END EPISODE TRACING CODE
    
    def get_best_A_for_S(self, S, actions):
        S = tuple(S)
        if self.DEBUG: print("get best A for ", str(self.sRep))
        max_acts = None
        max_val = -float("inf")
        #print(self.Q[S])
        #print([a.id for a in actions])
        if self.DEBUG: print(S)
#        if self.DEBUG: print([(a.id, self.Q[S][a]) for a in actions])
        for a in actions:
            aval = self.Q[S][a]                
            if aval > max_val:
                max_acts=[a]
                max_val=aval
            elif aval == max_val:
                max_acts.append(a)
                
        rand_from_max=random.choice(max_acts)
                
        if self.DEBUG:
            print("max acts ", [a.id for a in max_acts])
            print("new rand of max acts=", rand_from_max.id,"val=",max_val)
#             input("uq")
        #print("poss acts(", self.state_as_str(S),") = ", [(c.id, acts_dict[c]) for c in acts_dict])
        #max_key_by_val = max(acts_dict, key = acts_dict.get)
        return rand_from_max
    

    @abstractmethod
    def sa_update(self, *params):
        '''here is where you update the Q(nx_S,nx_A) values for a state-action pair'''
        pass


    def run_episode(self, model, stu, max_steps=-1, update_qvals=True, reset_student=False):
        #max_steps = float('inf') if max_steps<=0 else max_steps # hack up an infinite number of attempts here
        step_cnt=0
        
        #PUT YOUR LEARNING STUFF HERE
        
        return step_cnt
        
    def get_next_state(self,S,A, succ):
        cp = list(S)
        cp[A.id]=succ
        return tuple(cp)
    
    def extend_Q(self, S, actions):
        S = tuple(S)
        if S not in self.Q:
            self.Q[S]={}
            
        known_actions = self.Q[S]
        for c in actions:
            if c not in known_actions:
                print(c.id, "not a known action for state,",self.sRep,"..extending Q")
                self.Q[S][c]= -1.0
        if self.DEBUG: print("Extended Q for ", S)


    def get_next_lesson(self):
        lesson, ex = self.choose_A(self.sRep.S, self.possible_actions)
        return lesson, ex
    
    def record_lesson_results(self, success):
        pass
    
    def choose_A(self, S, actions=None):
        if self.Q=={}:
            explore = True
        elif self.EPS>0:
            explore = randint(0,self.EPS)==0
        else:
            explore = False

        if (explore): # starting move OR exploratory move
            a = random.choice(actions)
            if self.DEBUG: print("explore",a.id)
        else: #exploitative move
            a = self.get_best_A_for_S(S, actions)
            if self.DEBUG: print("exploit",a.id)        
        return a, explore
