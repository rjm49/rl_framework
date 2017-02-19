'''
Created on 30 Nov 2016

@author: Russell
'''
from random import randint
import random
from reinf.exp1.tutors.abstract import AbstractTutor
from reinf.exp1.policies.policy_utils import state_as_str
from reinf.exp1.classes import Concept
from _tracemalloc import get_traceback_limit
import copy
from reinf.exp1.domains.filterlist_utils import update_filter
from reinf.exp1.tutors.base import BaseTutor


class SarsaGoalTutor(BaseTutor):
    '''
    Tutoring agent with simple tabular state representation and SARSA type on-policy control algorithm
    '''


    def __init__(self, num_nodes=100, alpha=0.5, eps=0.01, gamma=1.0, name="SarsaTutor"):
        '''
        Constructor
        '''
        self.DEBUG=False
        #self.student_knowledge = [False for _ in range(num_nodes)]
        self.Q = {} #here live the { thisS: [actions] } pairs for each tabular thisS...
        self.EPS = eps
        self.learn_rate = alpha
        self.gamma = gamma
        self.num_nodes = num_nodes
                
        self.thisS = "X" # tuple([False] * num_nodes)
        self.lastS = ()
        self.lastA = None
        self.lastR = None
        
        self.filterlist = {}
        self.name = name
                
    def sa_update(self, S,A,R,nx_S,nx_A):
        Q0 = self.Q[S][A]
        Q = self.Q[nx_S][nx_A]
        Q1 = Q0 + self.learn_rate*((R + self.gamma*Q) - Q0)
        self.Q[S][A]=Q1
        #self.thisS = nx_S
        #self.lastA = nx_A

    def run_episode(self, model, stu, max_steps=-1, update_qvals=True):
        actions = model.concepts
        S = tuple([False] * len(actions)) #reset the tutor's state .. we assume the student knows nothing
        self.extend_Q(S, actions)
        #max_steps = float('inf') if max_steps<=0 else max_steps # hack up an infinite number of attempts here
        step_cnt=0
        
        lastA = None
        A, exp = self.choose_A(S, actions) # for SARSA we must pick the initial task outside the loop
        As = []
        Ss = []
        Qs = []
        msgs=[]
        #while step_cnt<=max_steps and not self.mission_complete():
        while (max_steps<=0 or step_cnt<=max_steps) and (False in S):
            #RECORD KEEPING
            As.append(A.id)
            Ss.append(state_as_str(S))
#             Qs.append( self.get_SA_val(S, A) )
            Qs.append( self.Q[S][A] )
            
            #INFERENCE
            if A.id not in self.filterlist:
                self.filterlist[A.id]= [True]*len(actions)
            
            #STATE UPDATE/Q VALS
            msgs.append("Attempt {} {} -> ? Q= {}".format( state_as_str(S), A.id, self.Q[S][A] ) )
#             R=-1.0
            print(self.name, end=" - ")
            succ = stu.try_learn(A)
            if succ:
                
                new_S = self.get_next_state(S,A)
                self.extend_Q(new_S, actions)
                
                R= -1.0
                if(False not in new_S):
                    R = 100.0 #basically if we've learned everything, get a big treat
                if self.DEBUG: print("success learned", A.id,"--> new S=",state_as_str(new_S))
                update_filter(self.filterlist, S, A.id, succ)# we use successful transitions as evidence to eliminate invalidly hypothesised dependencies
            else:
                new_S = S
                R= -1.0    
            
            msgs.append("{} {} -> {} Q={} R={} {} {}".format( state_as_str(S), A.id, state_as_str(new_S), self.Q[S][A], R, "S" if succ else "F", "X" if exp else "-" ) )
            
            new_A,exp = self.choose_A(new_S, actions)
            if(A==new_A):
                print("        *        Will try to Learn repeat lesson", A if A==None else A.id, "X" if exp else "-")
                        
            if update_qvals:
                self.sa_update(S, A, R, new_S, new_A)
            
#               self.lastS = self.thisS
            S = new_S
            lastA = A
            A = new_A
            step_cnt+=1
#             print(state_as_str(self.thisS), step_cnt)
        if step_cnt==max_steps:
            print("Terminated at step limit!")
        print("SARSA: Episode over in",step_cnt,"steps")
        if self.DEBUG: 
            for m in msgs:
                print(m)
#             print("States were:",Ss)
#             print("Actions were:",As)
#             print("Actions vals were:",Qs)
            print("Q(S,A) values:")
            for s in Ss:
                sb = tuple([bool(int(x)) for x in s])
                print(sb)
                for a in self.Q[sb]:
                    print(s, a.id, self.Q[sb][a])
                print("")

        return step_cnt
        