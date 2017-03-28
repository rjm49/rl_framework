'''
Created on 11 Nov 2016

@author: Russell
'''
import matplotlib.pyplot as plt
from reinf.exp1.domain_models import BranchMergeNetwork
from reinf.exp1.students.ideal import IdealStudent
from reinf.exp1.train_and_test import run_model
from reinf.exp1.tutors.random import RandomTutor
from reinf.exp1.tutors.tabular import SarsaTutor


training_length = 2000

num_models = 100
trials_per_model = 10
trial_length=450

num_missions=100
num_iter=float('inf')

N =100 # number of nodes
branch_factor = 3
epsilons = [-5]
alphas = [0.5]

batches = [
                {
                'run':True,
                'batch_name':'BMC3 - Random',
                'student': IdealStudent(),
                'model': BranchMergeNetwork(3),
                'tutor': "random",
#                 'Ks': [10*i for i in range(1,200)] + [1000*i for i in range(2,14)]
                'Ks': [i for i in range(1,10000,50)]
                },
             
                {
                'run':True,
                'batch_name':'BMC3 - Tabular',
                'student': IdealStudent(),
                'model': BranchMergeNetwork(3),
                'tutor': "tabular",
#                 'Ks': [10*i for i in range(1,200)] + [1000*i for i in range(2,14)]
                'Ks': [i for i in range(1,10000,50)]
                },

             ]


    
    
if __name__ == '__main__':
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
                    tutor = SarsaTutor(N, alpha, eps)
                    stu = IdealStudent()
                    tutor.train(models, stu, num_missions, num_iter)
                    print("done.")
                else:
                    tutor=RandomTutor(N)
                
                
                for model in models:
        #             gvrender(model)
                    model.name = "{} (a={}, e={})".format(batch_name,alpha,eps)                    
                    run_name, model_x, model_y = run_model(model, tutor, trials_per_model, trial_length, step_width=10)
                    
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
    leg = plt.legend(loc='lower right')
    leg.get_frame().set_alpha(0.3)
    plt.show()