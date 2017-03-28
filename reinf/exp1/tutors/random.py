'''
Created on 30 Jan 2017

@author: Russell
'''
import random
import copy
from reinf.exp1.policies.policy_utils import state_as_str
from reinf.exp1.domains.filterlist_utils import update_filter
from reinf.exp1.tutors.base import BaseTutor

class RandomTutor(BaseTutor):
    def __init__(self, num_nodes=100, alpha=None, epsilon=None, gamma=None, name='RandomTutor'):
        super().__init__()
        self.student_knowledge = [False for _ in range(num_nodes)]
        self.filterlist = {}
        self.num_nodes = num_nodes
        self.thisS = tuple([False] * num_nodes)
        self.name=name
        
    def reset(self):
        self.student_knowledge[:] = [False] * len(self.student_knowledge)
        self.thisS = tuple([False] * self.num_nodes)

    def mission_complete(self):
        return (False not in self.thisS)
    
    def update_state(self, A):
        if(not A): #i.e. A could be None in the starting state
            return
        self.student_knowledge[A.id]=True

    def student_tried(self, A, succ):
        if(not A):
            return
        self.student_knowledge[A.id]=True
        
    def choose_A(self,concepts):
        return random.choice(concepts)

    def run_episode(self, model, stu, max_steps=-1, update_qvals=True):
        self._new_trace()
        cs = model.concepts
        S = tuple([False for x in cs]) #reset the tutor's state .. we assume the student knows nothing
        if max_steps<0:
            max_steps = float('inf')
        step_cnt=0
        A = random.choice(cs) # pick the initial task
        
        while step_cnt<max_steps and not self.mission_complete():
            if A.id not in self.filterlist:
                self.filterlist[A.id]= [True]*len(cs)

            #print(self.name, end=" - ")
            succ = stu.try_learn(A)
            self._add_to_trace(S, A, succ)
            R=-1.0
            new_S = S
            if succ:
                R = -1.0
                new_S = self.get_next_state(A)
                update_filter(self.filterlist, S, A.id, succ)
            
            new_A = random.choice(cs)
#                     print(state_as_str(new_S))
#                     self.lastS = self.thisS
            S = new_S
            A = new_A
            step_cnt+=1
            #print(state_as_str(self.thisS), step_cnt)
        #print("RandomTutor: Episode over in",step_cnt,"steps")
        return step_cnt

    def train(self, models, stu, num_episodes=100, max_steps_per_ep=200, reset=True):
        print("random training")
        for model in models:
            mission_log = []
            for i in range(num_episodes):
                if(reset):
                    self.reset()
                    stu.reset_knowledge()
                #print("try again")
                k=1
                self.run_episode(model, stu, max_steps_per_ep, update_qvals=True)
                print("mission",i,"over in",k,"steps")
                #trial_score = len(stu.known_concepts)/len(dom.concepts) 
                #mission_log.append((i,trial_score))
                mission_log.append((i,k))
        return mission_log

    def get_next_state(self, A):
        cp = list(self.thisS)
        cp[A.id]=True
        return tuple(cp)

    def get_policy(self):
        return None
           
