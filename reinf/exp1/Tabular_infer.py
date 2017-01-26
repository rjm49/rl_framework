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
import codecs

num_models = 1
trials_per_model = 1
trial_length=2000

N = 100 # number of nodes
branch_factor = 5
epsilons = [3]
alphas = [0.5]

batches = [
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
    #create some chain of Concepts

#     model = FreeDomain()
#     model.regenerate(N)
    k_steps = 100
    master_log = []
    
    #create a single model upon which all our efforts will focus
    
    
    models = []
    for m in range(num_models):
        mod = BranchMergeNetwork(branch_factor)
        mod.regenerate(N)
        models.append(mod)
        gvrender(mod)
    
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
                tutor = TabularTutor(N, alpha, eps) # if batch["tutor"]=="tabular" else RandomTutor(N)
                student = IdealLearner()
                num_missions = 1000
#                 for model in models:
#                     model.name = "{} (a={}, e={})".format(batch_name,alpha,eps)                    
#                     run_name, model_x, model_y = run_model(model, tutor, alpha, eps, trials_per_model, trial_length, k_steps)                    
#                     if not batch_x:
#                         batch_x = model_x
#                         batch_y = [yv/num_models for yv in model_y]                    
#                     for i,score in enumerate(model_y):
#                         batch_y[i] = batch_y[i] + model_y[i]/num_models
#                 master_log.append((run_name, model_x, model_y))
                tutor = train_tutor(models, tutor, student, num_missions)
        
        f = codecs.open("tutor_out.txt","w")
       
        q_list = tutor.qs
        for k in q_list:            
            v = q_list[k]
            print(k.id if k else "none", end='')
            for vitem in v:
                print("---",vitem.id,v[vitem])
        f.close()
        
        
