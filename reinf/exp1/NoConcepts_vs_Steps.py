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
from random import randint
from scipy.interpolate.interpolate import spline
import numpy
import math
from scipy.optimize.minpack import curve_fit
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import reinf
from reinf.exp1.domain_models import DivNTree, ConNTree, BranchMergeNetwork,\
    FreeDomain, ChainDomain
from reinf.exp1.students.ideal import IdealStudent
from reinf.exp1.tutors.random import RandomTutor

#     log_dir = "..\\..\\compare_tree_model_logs\\"
#     log_dir ="..\\..\\forgetting_logs\\"
#     log_dir="..\\..\\bmc_only\\"
log_dir ="..\\..\\nvssteps_logs\\"
plot = True
avg = True

def main():
    ns = []
#     ns = [x for x in range(1,1001,50)]
#     branchfs = [2,3,4,5]
    for i in range(1,1012,10):
        ns.append(i)

    
    
    write = True

#     model_types=[BranchMergeNetwork]
#     model_types=[ ChainDomain, DivNTree,ConNTree,BranchMergeNetwork]
    model_type = ConNTree
    branchf = 10
    
    if write:
        logfile = open(os.path.join(log_dir,"RND"+ str(model_type).split(".")[-1][:-2] + str(branchf)+".log"),"w")
        print("writing:",str(logfile))
        #m = model_type(branch_factor=branchf)
    for n in ns:
        
        tut = RandomTutor(name="RND")
        
        step_cnt=0
        #             m.regenerate(n)
        runs = 100
        print(n)
        for i in range(runs):
            m = model_type(branch_factor=branchf)
            m.regenerate(n)
            node_num = len(m.concepts)
            tut.possible_actions= m.concepts
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
            if write and not avg:
                print("writing")
                logfile.write("{},{}\n".format(node_num, step_cnt))
                step_cnt = 0
                    
        if write and avg:
            #print("writing log file")
            logfile.write("{},{}\n".format(node_num, step_cnt/runs))
            print(step_cnt/runs)

    if write: logfile.close()
 
if __name__ == '__main__':
    if not plot:
        main()
    else:
#         fnames=["RNDFreeDomain1.1.log","RNDChainDomain1.1.log", "RNDDivNTree10.log", "RNDDivNTree2.1.log", "RNDConNTree2.1.log", "RNDConNTree10.log", "RNDBranchMergeNetwork2x100.log", "RNDBranchMergeNetwork3.log", "RNDBranchMergeNetwork4.log", "RNDBranchMergeNetwork10.log"]
        fnames=["RNDFreeDomain1.1.log","RNDChainDomain1.1.log", "RNDDivNTree2.1.log", "RNDConNTree2.1.log", "RNDBranchMergeNetwork2x100.log"] #, "RNDBranchMergeNetwork3.log", "RNDBranchMergeNetwork4.log", "RNDBranchMergeNetwork10.log"]
        plotnames=["AC","CH","D2","C2","BMC2"] #, "BM2", "BM3", "BM4", "BM10"]
        for fname,plotname in zip(fnames, plotnames):
            freader  = csv.reader(open(os.path.join(log_dir,fname)), delimiter=',')
            steps=[]
            domain_size=[]
            print(fname)
            for row in freader:
                domain_size.append(int(row[0]))
                steps.append(float(row[1]))

            if "Branch" in fname:
#             if True:
                interval = len(domain_size)/len(set(domain_size))
                joind = zip(domain_size, steps)
                totals=defaultdict(float)
                maxs=defaultdict(float)
                mins=defaultdict(float)
                for j in joind:
                    totals[j[0]]+=j[1]/interval
                    if maxs[j[0]]<j[1] : maxs[j[0]]=j[1];
                    if mins[j[0]]==0 or mins[j[0]]>j[1] : mins[j[0]]=j[1];
                
                dszs= sorted(list(totals.keys()))
                avgtots= [ totals[x] for x in dszs ]
                maxtots= [ maxs[x] for x in dszs ]
                mintots= [ mins[x] for x in dszs ]
                
#                 pyplot.scatter(domain_size, steps, color="lightgrey", marker="o", s=0.5, label=plotname+" (Scatter)")
#                 pyplot.plot(dszs, avgtots, color="purple", label="BM2 (Average)")
#                 pyplot.plot(dszs, maxtots, color="purple")
#                 pyplot.plot(dszs, mintots, color="purple")
                
                powerlaw = lambda x, amp, index: amp * (x**index)
                #explaw = lambda x, a, b, c:  c + a*numpy.exp(b*x)
                def explaw(x, a,b,c):
#                     print("explaw:",a,b,c)
                    return (a * b**(x/c)) -a


#                calculate polynomial
#                 z = numpy.polyfit(domain_size, steps, 3)
#                 polylaw = numpy.poly1d(z)
                
                # calculate new x's and y's
                x_new = numpy.linspace(min(domain_size), max(domain_size), 50)
#                 y_new = f(x_new)

                bestR2 = 0
                bestlaw = None
                bestpms = None
                for law in [powerlaw]: #, explaw]:
                    popt,pcov = curve_fit(law, domain_size, steps)
                    test_ys = law(domain_size, *popt)
                    ss_res = numpy.sum((steps - test_ys) ** 2)# residual sum of squares
                    ss_tot = numpy.sum((steps - numpy.mean(steps)) ** 2)# total sum of squares
                    r2 = 1 - (ss_res / ss_tot)
                    
                    print(r2)
                    print(*popt)
                    if r2>bestR2:
                        bestR2 = r2
                        bestlaw = law
                        bestpms = popt

                print("best law is ", str(bestlaw))
                print("best params =", popt)

                maxpopt,pcov = curve_fit(bestlaw, dszs, maxtots)
                minpopt,pcov = curve_fit(bestlaw, dszs, mintots)

                pyplot.plot(x_new, bestlaw(x_new, *bestpms), label=plotname)
#                 pyplot.plot(x_new, bestlaw(x_new, *maxpopt), ls="--", color="purple", label="BM2 (Max|Min)")
#                 pyplot.plot(x_new, bestlaw(x_new, *minpopt), ls="--", color="purple")
            else:
#                 pass
                pyplot.plot(domain_size, steps, label=plotname)
        leg = pyplot.legend(loc='upper left')
        leg.get_frame().set_alpha(0.1)
        pyplot.title("Domain Size vs Difficulty (Ideal Student, Random Tutor, k=2)")
        pyplot.xlabel("Domain Size (#Concepts to Learn)")
        pyplot.ylabel("Difficulty (#Steps to Master)")
        pyplot.xlim(0,450)
        pyplot.ylim(0,200000)
        pyplot.show()
