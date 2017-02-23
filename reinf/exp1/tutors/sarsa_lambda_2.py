'''
Created on 30 Nov 2016

@author: Russell
'''
from reinf.exp1.policies.policy_utils import state_as_str
from reinf.exp1.tutors.base import BaseTutor


class SarsaL2(BaseTutor):
    '''
    Tutoring agent with simple tabular state representation and SARSA type on-policy control algorithm, with Eligibility Trace (Lambda)
    '''


    def __init__(self, num_nodes=100, alpha=0.5, eps=0.01, gamma=1.0, name="SarsaL"):
        '''
        Constructor
        '''
        super().__init__(num_nodes, alpha, eps, gamma, name)
#         self.DEBUG=False
#         #self.student_knowledge = [False for _ in range(num_nodes)]
#         self.Q = {} #here live the { thisS: [actions] } pairs for each tabular thisS...
#         self.EPS = eps
#         self.learn_rate = alpha
#         self.gamma = gamma
#         self.num_nodes = num_nodes
#                 
#         self.thisS = "X" # tuple([False] * num_nodes)
#         self.lastS = ()
#         self.lastA = None
#         self.lastR = None
#         
#         self.filterlist = {}
        self.name = name
        
        self.history = []
        self.lambda_val = 0.0
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
#         input("history limit is "+ str(self.history_limit))
                
    def sa_update(self, S,A,R,nx_S,nx_A):
        Q = self.Q[S][A]
        nx_Q = self.Q[nx_S][nx_A]
        delta = R + self.gamma*nx_Q - Q
        learnage = self.learn_rate * delta
        decay = self.gamma * self.lambda_val
        z_credit = 1.0
        seen=set()
        for sa in self.history[:self.history_limit]:
            if tuple(sa) not in seen:
                s = sa[0]
                a = sa[1]
                self.Q[s][a] += learnage*z_credit
                seen.add(tuple(sa))
            z_credit = z_credit * decay
        

    def run_episode(self, model, stu, max_steps=-1, update_qvals=True):
        self._new_trace()
        actions = model.concepts
        S = tuple([False] * len(actions)) #reset the tutor's state .. we assume the student knows nothing
        self.extend_Q(S, actions)
        #max_steps = float('inf') if max_steps<=0 else max_steps # hack up an infinite number of attempts here
        step_cnt=0
        

        A, exp = self.choose_A(S, actions) # for SARSA we must pick the initial task outside the loop
        As = []
        Ss = []
        Qs = []
        msgs=[]
        self.history=[]
        #while step_cnt<=max_steps and not self.mission_complete():
        while (max_steps<=0 or step_cnt<=max_steps) and (False in S):
            self.history = [[S,A]] + self.history
#             if self.history > self.history_limit:
#                 self.history.pop()
            
            #RECORD KEEPING
            As.append(A.id)
            Ss.append(state_as_str(S))

            Qs.append( self.Q[S][A] )
            
            #INFERENCE
#             if A.id not in self.filterlist:
#                 self.filterlist[A.id]= [True]*len(actions)
            
            #STATE UPDATE/Q VALS
            msgs.append("Attempt {} {} -> ? Q= {}".format( state_as_str(S), A.id, self.Q[S][A] ) )

            print(self.name, end=" - ")
            succ = stu.try_learn(A)
            self._add_to_trace(S, A, succ)
            if succ:
                
                new_S = self.get_next_state(S,A)
                self.extend_Q(new_S, actions)
                
                R= -1.0
                if(False not in new_S):
                    R = 100.0 #basically if we've learned everything, get a big treat
                if self.DEBUG: print("success learned", A.id,"--> new S=",state_as_str(new_S))
#                 update_filter(self.filterlist, S, A.id, succ)# we use successful transitions as evidence to eliminate invalidly hypothesised dependencies
            else:
                new_S = S
                R= -1.0    
            
            msgs.append("{} {} -> {} Q={} R={} {} {}".format( state_as_str(S), A.id, state_as_str(new_S), self.Q[S][A], R, "S" if succ else "F", "X" if exp else "-" ) )
            
            new_A,exp = self.choose_A(new_S, actions)
            if(A==new_A):
                print("        *        Will try to Learn repeat lesson", A if A==None else A.id, "X" if exp else "-")
                        
            if update_qvals:
                self.sa_update(S, A, R, new_S, new_A)
            
            S = new_S
            A = new_A
            step_cnt+=1

        if step_cnt==max_steps:
            print("Terminated at step limit!")
        print("SARSA-L: Episode over in",step_cnt,"steps")
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
        