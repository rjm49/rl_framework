'''
Created on 24 Jan 2017

@author: Russell
'''
import sys
from reinf.exp1.IdealLearner import IdealLearner


def run_model(model, tutor, alpha, epsilon, trials_per_model, steps_per_trial, step_width):
    model_x = []
    model_y = []
    run_name = model.name # "{} a={} e={}".format(model.name, alpha, epsilon)
    sys.stdout.write(run_name)
    for trial in range(trials_per_model):
        #sys.stdout.write("\r{}".format(trial))
        student = IdealLearner()
        tutor.reset_steps()
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
        ct = tut.pick_another(dom.concepts)
#       print("concept picked=",ct.id)
        tut.steps += 1
        succ = stu.try_learn(ct)
        if succ:
#           print("learned",ct.id,"successfully")
            tut.student_learned(ct, k)       
#             print("".join(['X' if stu.knows(n) else "-" for n in dom.concepts]))
#                 print([(c.id,[p.id for p in c.predecessors]) for c in model.concepts])

def train_tutor(dom_array, tut, stu, num_missions=1000):
    for dom in dom_array:
        for i in range(num_missions):
            tut.reset_steps()
            tut.reset_knowledge()
            stu.reset_knowledge()
            print("try again")
            while not tut.mission_complete():
                learn_k_steps(1, tut, stu, dom)
            print("mission",i,"complete!")
    return tut