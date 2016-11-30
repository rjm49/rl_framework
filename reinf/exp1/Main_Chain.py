'''
Created on 11 Nov 2016

@author: Russell
'''
from reinf.exp1.classes import Concept
import random
import matplotlib.pyplot as plt
from reinf.exp1.IdealLearner import IdealLearner
from reinf.exp1.domain_models import ChainDomain, FreeDomain, Div2Tree, Con2Tree,\
    BranchMergeNetwork
from matplotlib import legend

def printNode(c):
    print(c.id)
    for p in c.predecessors:
        printNode(p)


N = 100 # number of nodes

num_trials = 100

scores = []
K_vals = []

batches = [
           {
                'run':True,
                'batch_name':'free',
                'learner': IdealLearner(),
                'domain': FreeDomain(),
#                 'Ks': [10*i for i in range(1,100)] + [1000*i for i in range(1,14)]
                'Ks': [i for i in range(1,10000,50)]
            },
           {
                'run':True,
                'batch_name':'chained',
                'learner': IdealLearner(),
                'domain': ChainDomain(),
#                 'Ks': [100*i for i in range(1,10)] + [1000*i for i in range(1,14)]
                'Ks': [i for i in range(1,10000,50)]

            },
            {
                'run':True,
                'batch_name':'div\'t_bin_tree',
                'learner': IdealLearner(),
                'domain': Div2Tree(),
#                 'Ks': [10*i for i in range(1,200)] + [1000*i for i in range(2,14)]
                'Ks': [i for i in range(1,10000,50)]
            },
            {
                'run':True,
                'batch_name':'conv\'t_bin_tree',
                'learner': IdealLearner(),
                'domain': Con2Tree(),
#                 'Ks': [10*i for i in range(1,200)] + [1000*i for i in range(2,14)]
                'Ks': [i for i in range(1,10000,50)]
            },
            {
                'run':True,
                'batch_name':'branch/merge/chain',
                'learner': IdealLearner(),
                'domain': BranchMergeNetwork(),
#                 'Ks': [10*i for i in range(1,200)] + [1000*i for i in range(2,14)]
                'Ks': [i for i in range(1,10000,50)]
            }
           ]

if __name__ == '__main__':
    #create some chain of Concepts

#     domain = FreeDomain()
#     domain.regenerate(N)

    master_log = []
    
    for batch in batches:
        
        if not batch["run"]:
            continue
        
        batch_msgs=[]
        batch_x=[]
        batch_y=[]
        name=batch["batch_name"]
        print(name, '...')
        domain = batch["domain"]
        #Ks = batch["Ks"]
        #Ks = [i for i in range(1,2000,10)] + [i for i in range(2000,15000,500)]
        
        Ks = [i for i in range(1,15000,500)]
        
        for K in Ks:
            print(K)
            ave_score = 0.0
            for _ in range(num_trials):
                #we begin in a state of blissful ignorance
                domain.regenerate(N)
                learner = IdealLearner()
                #and then by random chance the horror of life unfolds...
                for _ in range(K):
                    ct = random.choice(domain.concepts)
                    learner.try_learn(ct)
                
#                 print("".join(['X' if learner.knows(n) else "-" for n in domain.concepts]))
#                 print([(c.id,[p.id for p in c.predecessors]) for c in domain.concepts])
                
                score = len(learner.known_concepts)/len(domain.concepts)
#                 print("Score for", K, "was:", score)    
                ave_score += float(1/num_trials) * score
         
            #batch_msgs.append("%s: av.score w %d learning steps (across %d trials) = %f" % (name, K, num_trials, ave_score))
        
            batch_x.append(K)
            batch_y.append(ave_score)
        
        master_log.append((name, batch_x, batch_y, batch_msgs))
    
    #    print("Average score with ",K,"learning steps (across",num_trials,"trials =",ave_score)
    for s in master_log:
#         print(s[3])
        plt.plot(s[1], s[2], label=s[0])
    
    plt.ylabel('ave score (%s concepts, %s trials)' % (N, num_trials))
    plt.xlabel('learning iterations')
    leg = plt.legend(loc='right')
    leg.get_frame().set_alpha(0.3)
    plt.show()