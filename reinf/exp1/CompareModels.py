'''
Created on 7 Feb 2017

@author: Russell
'''
from _collections import defaultdict
import os

from reinf.exp1.domain_models import DivNTree, ConNTree, BranchMergeNetwork,\
    ChainDomain, FreeDomain
from reinf.exp1.students.ideal import IdealStudent
from reinf.exp1.tutors.random import RandomTutor
from reinf.viz.gviz import gvrender
from reinf.exp1.students.forgetting import ForgettingStudent
from reinf.exp1.students.relearning import RelearningStudent


def main():

    max_steps=12000
    n = 100
#     branchfs = [2,3,4,5]
    branchfs = [2]
    
#     log_dir = "..\\..\\compare_tree_model_logs\\"
#     log_dir ="..\\..\\forgetting_logs\\"
#     log_dir="..\\..\\bmc_only\\"
    log_dir ="..\\..\\relearning_logs\\"
    
    write = True

    model_types=[BranchMergeNetwork]
#     model_types=[ ChainDomain, DivNTree,ConNTree,BranchMergeNetwork]
    #model_types = [FreeDomain, ChainDomain]
    
    models =[]
    for m in model_types:
        for b in branchfs:
            mod = m(branch_factor=b)
            mod.regenerate(n)
            models.append(mod)
            print("created "+str(mod))
            gvrender(mod, os.path.join(log_dir, str(type(mod)).split(".")[-1][:-2]+str(b) ))
    
    tut = RandomTutor()
    N=100
    
    for m in models:
        if write: logfile = open(os.path.join(log_dir,tut.name.split(" ")[0]+ str(type(m)).split(".")[-1][:-2] + str(m.branch_factor)+".log"),"w")
        
        m_scores = defaultdict(int)
        m_scores[0] = 0
        for i in range(N):
            step_cnt=0
            p = RelearningStudent()
#             p = ForgettingStudent()
#             p= IdealStudent()
            mod.regenerate(n)
            while step_cnt < max_steps:        
                ep_len = tut.run_episode(m, p, max_steps=1, update_qvals=True)        
                tut.transition_trace.clear() # clear this trace
                
                mastery = p.get_mastery_score()/len(m.concepts)

                step_cnt += ep_len
                m_scores[step_cnt]+=mastery/N
                
            pc=100.0*(i+1)/N
            if (pc == int(pc)):
                print("{}%".format(pc))

        for step_cnt in sorted(m_scores.keys()):            
            mastery = m_scores[step_cnt]
            if write:
                #print("writing log file")
                logfile.write("{},{}\n".format(step_cnt, mastery))

        if write: logfile.close()
 
if __name__ == '__main__':
    main()