'''
Created on 30 Nov 2016

@author: Russell
'''
from reinf.exp1.policies.policy_utils import state_as_str
from reinf.exp1.domains.filterlist_utils import update_filter
from reinf.exp1.tutors.base import BaseTutor
from _collections import defaultdict
from profile_decr import profile
import codecs

class SarsaLambdaTutor(BaseTutor):
    '''
    Tutoring agent with simple tabular state representation and SARSA type on-policy control algorithm
    '''
    def __init__(self, num_nodes=100, alpha=0.5, eps=0.01, gamma=1.0, name="SarsaLambdaTutor"):
        '''
        Constructor
        '''
        self.DEBUG=False
        #self.student_knowledge = [False for _ in range(num_nodes)]
        self.Q = {} #here live the { thisS: [actions] } pairs for each tabular thisS...
        self.EPS = eps
        self.learn_rate = alpha
        self.gamma = gamma
        self.lambda_val=0.0
#         self.num_nodes = num_nodes
#         
        self.filterlist = {}
        self.name = name
        
        self.valid_states=set()
        self.logfile = codecs.open("sarsaLambda.log", "w")
        self.recent_SAs=[]
        self.hustory_limit=10
        
        
        gxl=self.gamma*self.lambda_val
        if self.lambda_val==0:
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
#         input("history limit is "+ str(self.history_limit))
                
#     def extend_Z(self, Z, S, actions):
#         if S not in Z:
#             Z[S]={}
#             for c in actions:
#                 Z[S][c]=0.0
#             if self.DEBUG: print("extended Z for ", state_as_str(S))

#     @profile
    def sa_update(self, S,A,R,nx_S,nx_A, valid_states, all_actions, msgs):
        nx_Q = self.Q[nx_S][nx_A]
        Q = self.Q[S][A] if A else 0.0
        delta = R + self.gamma*nx_Q - Q
#        dels.append(delta)
        scaled_delta = self.learn_rate*delta
        z_decay = self.gamma * self.lambda_val
#         Z[S][A] = 1.0 #use a replacement trace because we have long learning paths
        cell_counter=0
        z=1.0
        for sa in self.recent_SAs: #valid states is an approximation of valid game states found by experience, gets better over time
            s = sa[0]
            a = sa[1]
            self.Q[s][a] = scaled_delta*z
            z *= z_decay
            #print(state_as_str(s),a.id,":= (",scaled_delta,")*",z)
            cell_counter+=1
        print("updated {} cells".format(cell_counter))
        
        self.logfile.write("update to " + state_as_str(S))
        self.logfile.write("delta= "+str(delta)+", new Q,Z vals:")
#         for s in valid_states:
#             Q_s = self.Q[s]
#             for a in Q_s:
#                 #msgs.append("{} {} {} {}".format(state_as_str(s), a.id, Q_s[a], Z_s[a]))
#                 self.logfile.write(state_as_str(s)+ " " +str(a.id)+ " " +str(Q_s[a]))
#             self.logfile.write("~ ~ ~ ~")

#     @profile
    def run_episode(self, model, stu, max_steps=-1, update_qvals=True):
        actions = model.concepts
                
        S = tuple([False] * len(actions)) #reset the tutor's state .. we assume the student knows nothing        
        self.extend_Q(S, actions)
#         self.extend_Z(Z, S, actions)
        self.valid_states.add(S)
        #max_steps = float('inf') if max_steps<=0 else max_steps # hack up an infinite number of attempts here
        step_cnt=0
        
        lastA = None
        A, exp = self.choose_A(S, actions) # for SARSA we must pick the initial task outside the loop
        As = []
        Ss = []
        Qs = []
        dels=["-"]
        msgs=None
        #while step_cnt<=max_steps and not self.mission_complete():
        while (max_steps<=0 or step_cnt<=max_steps) and (False in S):
            self.recent_SAs = [[S,A]] + self.recent_SAs
            if(len(self.recent_SAs))>self.history_limit:
                self.recent_SAs.pop()

            #INFERENCE
            if A.id not in self.filterlist:
                self.filterlist[A.id]= [True]*len(actions)
            
            #STATE UPDATE/Q VALS
            succ = stu.try_learn(A)
            if succ:
                new_S = self.get_next_state(S,A)
                
                self.valid_states.add(new_S)
                self.extend_Q(new_S, actions)
                R= -1.0
                if(False not in new_S):
                    R = 100.0 #basically if we've learned everything, get a big treat
                if self.DEBUG: print("success learned", A.id,"--> new S=",state_as_str(new_S))
                update_filter(self.filterlist, S, A.id, succ)# we use successful transitions as evidence to eliminate invalidly hypothesised dependencies
            else:
                new_S = S
                R= -1.0    
            
            self.logfile.write("{} {} -> {} (Q {} R {} {} {})\n".format( state_as_str(S), A.id, state_as_str(new_S), self.Q[S][A], R, "S" if succ else "F", "X" if exp else "-" ) )
            
            new_A,exp = self.choose_A(new_S, actions)
            if(A==new_A):
                print("        *        Will try to Learn repeat lesson", A if A==None else A.id, "X" if exp else "-")
                        
            if update_qvals:
#                 print("update...")
                self.sa_update(S, A, R, new_S, new_A, self.valid_states, actions, msgs)
#                 print("..done")
#               self.lastS = self.thisS
            S = new_S
            lastA = A
            A = new_A
            step_cnt+=1
#             print(state_as_str(self.thisS), step_cnt)
        if step_cnt==max_steps:
            print("Terminated at step limit!")
        print("SARSA: Episode over in",step_cnt,"steps")

        return step_cnt
        