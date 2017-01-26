'''
Created on 11 Nov 2016

@author: Russell
'''
import matplotlib.pyplot as plt
from reinf.exp1.IdealLearner import IdealLearner
from reinf.exp1.domain_models import BranchMergeNetwork
from reinf.exp1.tabular_tutor import TabularTutor, RandomTutor
from reinf.viz.gviz import gviz_representation, gvrender
import sys
from reinf.exp1.train_and_test import learn_k_steps, run_model, train_tutor

def printNode(c):
    print(c.id)
    for p in c.predecessors:
        printNode(p)



training_length = 1000

num_models = 1
trials_per_model = 10
trial_length=5000

N = 100 # number of nodes
branch_factor = 3
epsilons = [3.0]
alphas = [0.5]

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


    
    
if __name__ == '__main__':
    k_steps = 10
    master_log = []
    
    models = []
    for m in range(num_models):
        mod = BranchMergeNetwork(branch_factor)
        mod.regenerate(N)
        models.append(mod)
    
    for batch in batches:
        if not batch["run"]:
            continue
        batch_name=batch["batch_name"]

        for alpha in alphas:
            for eps in epsilons:
                batch_y=[]
                batch_x=[]
                batch_msgs=[]
                batch_score=[]
                
                tutor=None
                if batch["tutor"]=="tabular":
                    print("training tutor")
                    tutor = TabularTutor(N, alpha, eps)
                    stu = IdealLearner()
                    train_tutor(models, tutor, stu, training_length)
                    print("done.")
                else:
                    tutor=RandomTutor(N)
                
                
                for model in models:
        #             gvrender(model)
                    model.name = "{} (a={}, e={})".format(batch_name,alpha,eps)                    
                    run_name, model_x, model_y = run_model(model, tutor, alpha, eps, trials_per_model, trial_length, k_steps)
                    
                    if not batch_x:
                        batch_x = model_x
                        batch_y = [yv/num_models for yv in model_y]
                    
                    for i,score in enumerate(model_y):
                        batch_y[i] = batch_y[i] + model_y[i]/num_models
                                        
                master_log.append((run_name, model_x, model_y))

    #    print("Average score with ",K,"learning steps (across",trials_per_model,"trials =",ave_score)
    for s in master_log:
#         print(s[3])
        plt.plot(s[1], s[2], label=s[0])
    
    plt.ylabel('ave score ({} models of {} concepts, {} trials each)'.format(num_models, N, trials_per_model))
    plt.xlabel('learning iterations')
    leg = plt.legend(loc='right')
    leg.get_frame().set_alpha(0.3)
    plt.show()