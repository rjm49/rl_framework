'''
Created on 11 Nov 2016

@author: Russell
'''
import codecs

from reinf.exp1.domain_models import BranchMergeNetwork, Domain
from reinf.exp1.domains.domain_utils import load_concepts_from_file,\
    score_domain_similarity
from reinf.exp1.policies.policy_utils import save_policy, state_as_str
from reinf.exp1.students.ideal import IdealLearner
from reinf.exp1.train_and_test import run_greedy
from reinf.exp1.tutors.sarsa import SarsaTutor
from reinf.viz.gviz import gvrender, gviz_representation
import matplotlib.pyplot as plt
from reinf.exp1.tutors.qutor import Qutor
from reinf.exp1.classes import Concept
from reinf.exp1.domains.filterlist_utils import clean_filterlist


num_models = 1
trials_per_model = 1

NO_MAX=float('inf')
num_missions = 8000

N = 40# number of nodes
branch_factor = 1
epsilons = [ 4 ]
alphas = [0.1]


batches=[1,500,1000,3000,5000]


if __name__ == '__main__':
    #create some chain of Concepts
    
    #create a single model upon which all our efforts will focus
    
    

    mod = BranchMergeNetwork(branch_factor)
    mod.regenerate(N)
    saved_struct = load_concepts_from_file("test40.dat")    
    mod.concepts = saved_struct
    gvrender(mod, "real")
    models = [mod]
    
    train_logs = []
    batch_names =[]
    for num_missions in batches:

        for alpha in alphas:
            for eps in epsilons:
                tutor = SarsaTutor(N, alpha, eps) # if batch["tutor"]=="tabular" else RandomTutor(N)
                student = IdealLearner()
                log = tutor.train(models, student, num_missions, NO_MAX)
                
        
#         f = codecs.open("train_log.txt","w")
#         for L in log:
#             f.write(str(L)+"\n")
#         f.close()

        #save_policy(tutor.qs, "policy_file.dat")

        tutor.filterlist = clean_filterlist(tutor.filterlist)

        dummod = Domain()
        dummod.concepts = [Concept(i) for i,c in enumerate(mod.concepts) ]
                
        for k,v in tutor.filterlist.items():
            print(k, state_as_str(v))
            con = dummod.concepts[k]
            pixs = [ix for ix,bl in enumerate(v) if bl ] #get ids where state entry is True
            for i in pixs:
                con.predecessors.append(dummod.concepts[i])            
        gvrender(dummod, "inferred"+str(num_missions))

        err = score_domain_similarity(mod, dummod)
        train_logs.append([num_missions, err])
        #input("hit key")

        for i,model in enumerate(models):
            print("greed run",i)
            for _ in range(10):
                run_greedy(model, tutor)

        #         print(s[3])
    plt.ylabel('Steps to learn {} concepts (BMC{} Domain)'.format(N, branch_factor))
    plt.xlabel('Mission #')
    batch_name, num_missions, err = zip(*train_logs)
#         no, score = zip(*log)
    plt.plot(num_missions, err, label="SARSA Tutor")
    leg = plt.legend(loc='right')
    leg.get_frame().set_alpha(0.3)
    plt.show()
        
