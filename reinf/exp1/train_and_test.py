'''
Created on 24 Jan 2017

@author: Russell
'''
import sys
from reinf.exp1.students.ideal import IdealLearner
from reinf.exp1.tutors.sarsa import SarsaTutor as tutor
from reinf.exp1.policies.policy_utils import state_as_str, qvals_to_policy
from reinf.exp1.classes import Concept
from reinf.viz.gviz import gviz_representation


def run_greedy(model, tutor):
    A_list = []
    tutor.reset()
    tutor.EPS=0 #set the tutor to greedy policy mode
    while False in tutor.thisS:
        #print("from",state_as_str(tutor.thisS))
        A = tutor.choose_A(model.concepts)
        tutor.thisS = tutor.get_next_state(A)
        #print("chose",A.id if A else -1,"new state",state_as_str(tutor.thisS))
        A_list.append(A) # just make a list of all the actions we took
        #input("h")
    print([a.id for a in A_list])
    return A_list

def run_model(model, tutor, trials_per_model, steps_per_trial, step_width=1):
    model_x = []
    model_y = []
    run_name = model.name # "{} a={} e={}".format(model.name, alpha, epsilon)
    sys.stdout.write(run_name)
    for trial in range(trials_per_model):
        #sys.stdout.write("\r{}".format(trial))
        student = IdealLearner()
        tutor.reset()
        i=0
        steps = 0
        while steps < steps_per_trial:
            learn_k_steps(step_width, tutor, student, model)
            trial_score = len(student.known_concepts)/len(model.concepts)
            
            if(steps not in model_x):
                model_x.append(steps)
                model_y.append(0)
                
            model_y[i]+=trial_score/(trials_per_model)
            steps += step_width
            i += 1
    sys.stdout.write("..done!\n")
    return run_name, model_x, model_y

def learn_k_steps(K,tut,stu,dom):
    for k in range(K):
        act = tut.choose_A(dom.concepts)
        succ = stu.try_learn(act)
        if succ:
            tut.update_state(act)


def test_tutor(dom_array, tut, stu, num_missions=100, num_iter=100):
    for dom in dom_array:
        mission_log = []
        for i in range(num_missions):
            tut.reset()
            stu.reset_knowledge()
            #print("try again")
            k=1
            act = tut.choose_A(dom.concepts) # pick the initial task
            #print("first ACtion =", act.id)
            tut.lastA = act
            tut.lastS = tut.thisS
            while k<=num_iter and not tut.mission_complete():
                succ = stu.try_learn(act)
                R = 1.0 if succ else 0.0
                print(tut.state_as_str(tut.thisS), ": stu tried to learn",act.id,"with succ:",succ)
                if succ:
                    tut.update_state(act)
                #tut.student_tried(act, succ)
                act = tut.choose_A(dom.concepts)
                #print("new action=",act.id)
                tut.sa_update(tut.lastS, tut.lastA, R, tut.thisS, act)
                tut.lastS = tut.thisS
                tut.lastA = act
                k+=1
            print("mission",i,"over in",k,"steps")
            #trial_score = len(stu.known_concepts)/len(dom.concepts) 
            #mission_log.append((i,trial_score))
            mission_log.append((i,k))
    return tut, mission_log