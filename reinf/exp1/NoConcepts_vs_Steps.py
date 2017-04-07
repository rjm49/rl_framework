'''
Created on 7 Feb 2017

@author: Russell
'''
from _collections import defaultdict
import csv
from email._header_value_parser import Domain
import os

from matplotlib import pyplot

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import reinf
from reinf.exp1.domain_models import DivNTree, ConNTree, BranchMergeNetwork
from reinf.exp1.students.ideal import IdealStudent
from reinf.exp1.tutors.random import RandomTutor

#     log_dir = "..\\..\\compare_tree_model_logs\\"
#     log_dir ="..\\..\\forgetting_logs\\"
#     log_dir="..\\..\\bmc_only\\"
log_dir ="..\\..\\nvssteps_logs\\"

def main():

    ns = [x for x in range(1,200,4)]
#     branchfs = [2,3,4,5]
    branchf = 2
    
    
    write = True

#     model_types=[BranchMergeNetwork]
#     model_types=[ ChainDomain, DivNTree,ConNTree,BranchMergeNetwork]
    model_type = BranchMergeNetwork
    
    if write: logfile = open(os.path.join(log_dir,"RND"+ str(model_type).split(".")[-1][:-2] + str(branchf)+".log"),"w")
    m = model_type(branch_factor=branchf)
    for n in ns:
        
        print("new #nodes")
        m.regenerate(n)
        print("gen;d")
        node_num = len(m.concepts)
        tut = RandomTutor(name="RND")
        tut.possible_actions= m.concepts
        
        step_cnt=0
        
#             m.regenerate(n)
        runs = 10000
        print(node_num)
        for i in range(runs):
            m.regenerate(n)
#             print("reg'nd")
            p= IdealStudent()
            mastery = 0.0
            while mastery < 1.0:
#                 tut.test_student(p)
                step_cnt += 1
                lesson, ex = tut.get_next_lesson()
                success = p.try_learn(lesson)
                mastery = p.get_mastery_score()/len(m.concepts)
#                 print(node_num,mastery)
    #                 m_scores[step_cnt]=mastery #/num_training_sessions
        if write:
            #print("writing log file")
            logfile.write("{},{}\n".format(node_num, step_cnt/runs))
            print(step_cnt/runs)

#     if write: logfile.close()
 
if __name__ == '__main__':
    plot = False
    if not plot:
        main()
    else:
        fnames=["RNDFreeDomain1.log","RNDChainDomain1.log","RNDDivNTree2.log", "RNDConNTree2.log", "RNDBranchMergeNetwork2.log"]
        for fname in fnames:
            freader  = csv.reader(open(os.path.join(log_dir,fname)), delimiter=',')
            steps=[]
            domain_size=[]
            for row in freader:
                domain_size.append(int(row[0]))
                steps.append(float(row[1]))
            pyplot.plot(domain_size, steps, label=fname.split(".")[0])
        leg = pyplot.legend(loc='upper left')
        leg.get_frame().set_alpha(0.3)
        pyplot.show()
