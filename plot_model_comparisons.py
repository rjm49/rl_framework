'''
Created on 6 Mar 2017

@author: Russell
'''
import codecs
import re
from matplotlib import pyplot
from builtins import zip
import os

if __name__ == '__main__':
#     logdir = "bmc_only"
#     logdir = "compare_model_logs"
#     logdir = "forgetting_logs"
    logdir = "dynaqutor_logs"
    t_to_plot = ["DynaQutor2"]
    m_to_plot = ["BranchMergeNetwork"]
#     m_to_plot = ["ConNTree","DivNTree"]
#     m_to_plot = ["FreeDomain", "ChainDomain","ConNTree","DivNTree","BranchMergeNetwork"]
    #pure_names = ["RandomTutorFreeDomain1","RandomTutorChainDomain1"]
    pure_names=[]
#     b_to_plot = [2,3,4,5]
    b_to_plot=[2]
    #test_file = "itec2011"
    test_file =""
    fileext = ".log"
    

    pyplot.xlabel("# Lessons")
    pyplot.ylabel("Mastery")
    
    files_to_plot = pure_names + [t+m+str(b) for t in t_to_plot for m in m_to_plot for b in b_to_plot]
    print(files_to_plot)
    
    for t in files_to_plot:
        trace = []
        step_cnt = 0
        ep_len_cnt = 0
        filename = os.path.join(logdir, t+fileext);
        with open(filename) as f:
#             head = [next(f) for x in range(400)]
            head = f
            for line in head:
                #print(line.strip())
                #input("prompt")
                print(line)
                parts = line.strip().split(",")
                trace.append((int(parts[0]), float(parts[1])))
                #print(trace)
                
    
        steps, score = zip(*trace)
        pyplot.plot(steps, score, label=t)
        
        leg = pyplot.legend(loc='lower right')
        leg.get_frame().set_alpha(0.3)
    pyplot.show()