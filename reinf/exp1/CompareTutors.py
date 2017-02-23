'''
Created on 7 Feb 2017

@author: Russell
'''
from reinf.exp1.tutors.random import RandomTutor
from reinf.exp1.domain_models import BranchMergeNetwork
from reinf.exp1.domains.domain_utils import load_concepts_from_file,\
    score_domain_similarity, load_domain_model_from_file, save_domain_to_file
from reinf.viz.gviz import gvrender
from reinf.exp1.students.ideal import IdealLearner
from statistics import mean
from matplotlib import pyplot
from reinf.exp1.domains.filterlist_utils import build_inferred_model,\
    print_success_history_totals, intersect_all_history_totals
from reinf.exp1.tutors.sarsa_lambda_2 import SarsaL2
import tracemalloc
# tracemalloc.start()

def main():
#     epss=[2,100]
#     alphas=[0.1,0.9]
    epss=[5000]
#     alphas=[0.1, 0.5, 1.0]
    alphas=[1.0]
    gammas=[1.0] #discount factors
    lambdas=[0.0]
    #lambdas=[0.0, 0.3, 0.7, 0.99]
    max_episodes=1000
    plotting_interval=10
    
    load = True
    load_file="itec2011.dat"
    DEBUGG=False

    #LOAD The model
    if load:
        mod = load_domain_model_from_file(load_file) 
    else:
        mod = BranchMergeNetwork(2)
        mod.regenerate(5)
    gvrender(mod, "real")
#     save_domain_to_file(mod, "test10.dat")
    models = [mod]
    num_nodes = len(mod.concepts)

#     intervals=[x for x in range(1,500,10)]+[x for x in range(1,1001,100)]
    intervals=[x for x in range(1,max_episodes+1, plotting_interval)]

    tutorclasses=[
                    'RandomTutor',
#                     'SarsaGoalTutor',
#                     'QutorGoalTutor',
#                     'SarsaL2'
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
    
    paula = IdealLearner()

    fig, ax1 = pyplot.subplots()
    
    ax2 = ax1.twinx()
    ax1.set_xlabel('# of episodes')
    ax1.set_ylabel('Avg lessons to complete {}-node course, BF {}'.format( len(mod.concepts), mod.branch_factor ))
    ax2.set_ylabel('Error in inferred domain model')
 
#     snapshot1 = tracemalloc.take_snapshot()
    for tut in tutors:
        tut.DEBUG=DEBUGG
        episode_log= [] 
        inferr_log=[]   
        last_interv=0
        for interv in intervals:
            increm = interv - last_interv
            last_interv=interv
            episode_lengths=[]
            msgs=[]

            for j in range(increm):
                paula = IdealLearner()
                episode_lengths += [tut.run_episode(models[0], paula, max_steps=-1, update_qvals=True)]
                
            print(tutor,interv,episode_lengths)
#             snapshot2 = tracemalloc.take_snapshot()
#             top_stats = snapshot2.compare_to(snapshot1, 'lineno')
#             print("[ Top 10 differences ]")
#             for stat in top_stats[:10]:
#                 print(stat)
#             input("Hit return to continue..")
            
            m = mean(episode_lengths) if len(episode_lengths)>1 else episode_lengths[0]

            episode_log.append((interv, mean(episode_lengths)))            
            #MODEL INFERENCE CODE
            dummod = build_inferred_model(models[0], tut, interv)
            err = score_domain_similarity(models[0], dummod)
            inferr_log.append((interv, err))

#             if mean(episode_lengths) < 10:
#                 break
#         paula.reset_knowledge()
#         tut.EPS = 0 #set the tutor to greedy policy mode
#         kk = tut.run_episode(models[0], paula, max_steps=1000, update_qvals=False)
#         print("P-Learner took ",kk,"steps to learn about platypuses!")
    #         no, score = zip(*log)
        icnt, score = zip(*episode_log)
        ax1.plot(icnt, score, label=str(tut))        
        iv,er=zip(*inferr_log)
        ax2.plot(iv,er, label=str(tut)+"_inferr", linestyle='--')
        
        for i,tr in enumerate(tut.transition_trace):
            print("Episode trace",i,"for",tut)
            for j,step in enumerate(tr):
                print(j,step)
        
        intersect_all_history_totals(tut.transition_trace)

    leg = pyplot.legend(loc='upper right')
    leg.get_frame().set_alpha(0.3)
    pyplot.show()    

if __name__ == '__main__':
    main()