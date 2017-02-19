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

LOAD_FROM_FILE=False
num_models = 1
trials_per_model = 1

max_mission_length=float('inf')
num_missions = 8000

branch_factor = 2
epsilons = [ 4 ]
alphas = [0.5]

batches = [
                           {
                'run':True,
                'batch_name':'1',
                'num_missions':1
                },
                                      {
                'run':True,
                'batch_name':'0',
                'num_missions':500
                },

                {
                'run':True,
                'batch_name':'1000',
                'num_missions':1000
                },
                {'run':True, 'batch_name':'2000', 'num_missions':2000},
                {'run':True, 'batch_name':'4000 - Tabular', 'num_missions':4000},
                {'run':True, 'batch_name':'6000 - Tabular', 'num_missions':6000},
                {'run':True, 'batch_name':'7000 - Tabular', 'num_missions':7000},
                {'run':True, 'batch_name':'8000 - Tabular', 'num_missions':8000},
                {'run':True, 'batch_name':'8000 - Tabular', 'num_missions':16000},
             ]



if __name__ == '__main__':
    k_steps = 100
    master_log = []
   
    models = []
    N = 100# number of nodes
    for m in range(num_models):
        mod = BranchMergeNetwork(branch_factor)
        mod.regenerate(N)
        models.append(mod)
        gvrender(models[0], "real")

    if LOAD_FROM_FILE:
        saved_struct = load_concepts_from_file("test100.dat")    
        models[0].concepts = saved_struct
        gvrender(models[0], "real")

    tutor = SarsaTutor(N, 0.5, 5) # if batch["tutor"]=="tabular" else RandomTutor(N)
    student = IdealLearner()

    
    last_num=0
    main_log =[]
    for batch in batches:
        num_missions = batch['num_missions']
        more_missions = num_missions - last_num
        last_num= num_missions
        #                 tutor = Qutor(N, alpha, eps)
        log = tutor.train(models, student, more_missions, float("inf"))
        mnum, runlength =log[-1]
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
        main_log.append((num_missions, runlength, err))

    plt.ylabel('Steps to learn {} concepts (BMC{} Domain)'.format(N, branch_factor))
    plt.xlabel('Mission #')
    num_missions, runlength, err = zip(*main_log)
    
    print(num_missions)
    print(err)
    
#         no, score = zip(*log)
#     plt.plot(num_missions, runlength, label="Mission len")
    plt.plot(num_missions, err, label="Infer'nce err")
    leg = plt.legend(loc='right')
    leg.get_frame().set_alpha(0.3)
    plt.show()
        
