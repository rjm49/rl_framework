'''
Created on 7 Feb 2017

@author: Russell
'''
from reinf.exp1.tutors.random import RandomTutor
from reinf.exp1.domain_models import BranchMergeNetwork
from reinf.exp1.domains.domain_utils import _load_raw_from_file,\
    score_domain_similarity, load_domain_model_from_file, save_domain_to_file
from reinf.viz.gviz import gvrender
from reinf.exp1.students.ideal import IdealStudent
from statistics import mean
from matplotlib import pyplot
from reinf.exp1.domains.filterlist_utils import build_inferred_model,\
    print_success_history_totals, intersect_all_history_totals
from reinf.exp1.tutors.sarsa_lambda_2 import SarsaL2
from reinf.exp1.tutors.qutor import Qutor
from reinf.exp1.tutors.dynaqutor import DynaQutor
import tracemalloc
import codecs
import os
from reinf.exp1.students.forgetting import ForgettingStudent
# tracemalloc.start()

def main():
#     epss=[2,100]
#     alphas=[0.1,0.9]
    epss=[1000]
#     alphas=[0.1, 0.5, 1.0]
    alphas=[0.5]
    gammas=[1.0] #discount factors
    lambdas=[0.7]
    #lambdas=[0.0, 0.3, 0.7, 0.99]
    max_steps=100
    
    log_dir = "..\\..\\compare_logs\\"
    
    load = False
    load_file="itec2011.dat"
    DEBUGG=False
    write = True

    #LOAD The model
    if load:
        mod = load_domain_model_from_file(load_file) 
    else:
        mod = BranchMergeNetwork(4)
        mod.regenerate(100)
    gvrender(mod, "real")
#     save_domain_to_file(mod, "test10.dat")
    models = [mod]
    num_nodes = len(mod.concepts)

#     intervals=[x for x in range(1,500,10)]+[x for x in range(1,1001,100)]
#     intervals=[x for x in range(1,max_episodes+1, plotting_interval)]

    tutorclasses=[
                    'RandomTutor',
                    'Qutor',
                    'SarsaL2',
                    'DynaQutor'
                  ]
    
    tutors=[]
    for eps in epss:
        for alpha in alphas:
            for gamma in gammas:
                for lambduh in lambdas:
                    for classname in tutorclasses:
                        klass = eval(classname)
                        tutor = klass(num_nodes, alpha, eps, gamma, classname)
#                         if hasattr(tutor, "lambda_val"): #i.e. if this tutor uses an eligibility trace, it needs a decay value
                        
                        try:
                            tutor.lambda_val=lambduh
                            tutor.name+=(" L"+str(lambduh))
                        except AttributeError as aerr:
                            print(repr(aerr))
                        tutors.append(tutor)
    
#     fig, ax1 = pyplot.subplots()
    
#     ax2 = ax1.twinx()
#     ax1.set_xlabel('# of episodes')
#     ax1.set_ylabel('Avg lessons to complete {}-node course, BF {}'.format( len(mod.concepts), mod.branch_factor ))
#     ax2.set_ylabel('Error in inferred domain model')
 
    pyplot.xlabel("# episodes")
    pyplot.ylabel("Avg lessons to complete course")
 
#     snapshot1 = tracemalloc.take_snapshot()
    for tut in tutors:
        tut.DEBUG=DEBUGG
        if write: logfile = codecs.open(os.path.join(log_dir,tut.name.split(" ")[0]+".log"),"w")
        
        step_cnt=0
        while step_cnt < max_steps:        
            p = ForgettingStudent()
            ep_len = tut.run_episode(models[0], p, max_steps=-1, update_qvals=True)
            print(tutor, ep_len)

            if write:
                logfile.write("e\n")
                ep_t_log = tut.transition_trace.pop()
                for s in ep_t_log:
                    logfile.write(str(s)+"\n")
            step_cnt += ep_len

        if write: logfile.close()
        tut=None   

if __name__ == '__main__':
    main()