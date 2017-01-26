'''
Created on 11 Nov 2016

@author: Russell
'''
import random
import matplotlib.pyplot as plt
from reinf.exp1.IdealLearner import IdealLearner
from reinf.exp1.domain_models import ChainDomain, FreeDomain, Div2Tree, Con2Tree,\
    BranchMergeNetwork

def printNode(c):
    print(c.id)
    for p in c.predecessors:
        printNode(p)


N = 100 # number of nodes

trials_per_model = 1

scores = []
K_vals = []

batches = [
           {
                'run':True,
                'batch_name':'free',
                'student': IdealLearner(),
                'model': FreeDomain(),
#                 'Ks': [10*i for i in range(1,100)] + [1000*i for i in range(1,14)]
                'Ks': [i for i in range(1,10000,50)]
            },
           {
                'run':True,
                'batch_name':'chained',
                'student': IdealLearner(),
                'model': ChainDomain(),
#                 'Ks': [100*i for i in range(1,10)] + [1000*i for i in range(1,14)]
                'Ks': [i for i in range(1,10000,50)]

            },
            {
                'run':True,
                'batch_name':'div\'t_bin_tree',
                'student': IdealLearner(),
                'model': Div2Tree(),
#                 'Ks': [10*i for i in range(1,200)] + [1000*i for i in range(2,14)]
                'Ks': [i for i in range(1,10000,50)]
            },
            {
                'run':True,
                'batch_name':'conv\'t_bin_tree',
                'student': IdealLearner(),
                'model': Con2Tree(),
#                 'Ks': [10*i for i in range(1,200)] + [1000*i for i in range(2,14)]
                'Ks': [i for i in range(1,10000,50)]
            },
            {
                'run':True,
                'batch_name':'branch/merge/chain',
                'student': IdealLearner(),
                'model': BranchMergeNetwork(),
#                 'Ks': [10*i for i in range(1,200)] + [1000*i for i in range(2,14)]
                'Ks': [i for i in range(1,10000,50)]
            }
           ]

if __name__ == '__main__':
    #create some chain of Concepts

#     model = FreeDomain()
#     model.regenerate(N)

    master_log = []
    
    for batch in batches:
        
        if not batch["run"]:
            continue
        
        batch_msgs=[]
        batch_x=[]
        batch_y=[]
        name=batch["batch_name"]
        print(name, '...')
        model = batch["model"]
        #Ks = batch["Ks"]
        Ks = [i for i in range(1,2000,10)] #+ [i for i in range(2000,15000,500)]
        
        #Ks = [i for i in range(1,15000,500)]
        
        for K in Ks:
            print(K)
            ave_score = 0.0
            model.regenerate(N)
            for _ in range(trials_per_model):
                #we begin in a state of blissful ignorance
                student = IdealLearner()
                #and then by random chance the horror of life unfolds...
                for _ in range(K):
                    ct = random.choice(model.concepts)
                    student.try_learn(ct)
                
#                 print("".join(['X' if student.knows(n) else "-" for n in model.concepts]))
#                 print([(c.id,[p.id for p in c.predecessors]) for c in model.concepts])
                
                score = len(student.known_concepts)/len(model.concepts)
#                 print("Score for", K, "was:", score)    
                ave_score += float(1/trials_per_model) * score
         
            #batch_msgs.append("%s: av.score w %d learning steps (across %d trials) = %f" % (name, K, trials_per_model, ave_score))
        
            batch_x.append(K)
            batch_y.append(ave_score)
        
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