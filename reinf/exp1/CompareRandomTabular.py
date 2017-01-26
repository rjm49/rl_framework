'''
Created on 11 Nov 2016

@author: Russell
'''
import matplotlib.pyplot as plt
from reinf.exp1.IdealLearner import IdealLearner
from reinf.exp1.domain_models import BranchMergeNetwork
from reinf.exp1.tabular_tutor import TabularTutor, RandomTutor
from reinf.viz.gviz import gviz_representation, gvrender

def printNode(c):
    print(c.id)
    for p in c.predecessors:
        printNode(p)


N = 100 # number of nodes
trials_per_model = 1000

batches = [
                {
                'run':True,
                'batch_name':'BMC3 - Random',
                'student': IdealLearner(),
                'model': BranchMergeNetwork(3),
                'tutor': "random",
#                 'Ks': [10*i for i in range(1,200)] + [1000*i for i in range(2,14)]
                'Ks': [i for i in range(1,10000,50)]
                },
             
                {
                'run':True,
                'batch_name':'BMC3 - Tabular',
                'student': IdealLearner(),
                'model': BranchMergeNetwork(3),
                'tutor': "tabular",
#                 'Ks': [10*i for i in range(1,200)] + [1000*i for i in range(2,14)]
                'Ks': [i for i in range(1,10000,50)]
                },

             ]


def learn_k_steps(K,tut,stu,dom):
    for k in range(1,1+K):
        ct = tut.pick_another(dom.concepts)
#                     print("concept picked=",ct.id)
        tut.steps += 1
        succ = stu.try_learn(ct)
        if succ:
#                         print("learned",ct.id,"successfully")
            tut.student_learned(ct, k)       
#                 print("".join(['X' if student.knows(n) else "-" for n in model.concepts]))
#                 print([(c.id,[p.id for p in c.predecessors]) for c in model.concepts])
    
    
    
if __name__ == '__main__':
    #create some chain of Concepts

#     model = FreeDomain()
#     model.regenerate(N)
    k_steps = 5
    master_log = []
    
    #create a single model upon which all our efforts will focus
    model = BranchMergeNetwork(4)
    model.regenerate(N)
    gvrender(model)
    
    for batch in batches:
        if not batch["run"]:
            continue
        
        tutor = TabularTutor(N) if batch["tutor"]=="tabular" else RandomTutor(N)
        
        batch_msgs=[]
        batch_x=[]
        batch_y=[]
        batch_score=[]
        name=batch["batch_name"]

        for trial in range(trials_per_model):       
            print(name, "trial:", trial)
            
            student = IdealLearner()
            
            total_steps = 0
            i=0
            tutor.reset_steps()
            while total_steps < 5000:
                learn_k_steps(k_steps, tutor, student, model)
                trial_score = len(student.known_concepts)/len(model.concepts)
                
                if(total_steps not in batch_x):
                    batch_x.append(total_steps)
                    batch_y.append(0)
                    
                batch_y[i]+=trial_score/trials_per_model
                total_steps += k_steps
                i += 1
 
        master_log.append((name, batch_x, batch_y, batch_msgs))

    #    print("Average score with ",K,"learning steps (across",trials_per_model,"trials =",ave_score)
    for s in master_log:
#         print(s[3])
        plt.plot(s[1], s[2], label=s[0])
    
    plt.ylabel('ave score (%s concepts, %s trials)' % (N, trials_per_model))
    plt.xlabel('learning iterations')
    leg = plt.legend(loc='right')
    leg.get_frame().set_alpha(0.3)
    plt.show()