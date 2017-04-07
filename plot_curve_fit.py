'''
Created on 6 Mar 2017

@author: Russell
'''
import codecs
import re
from matplotlib import pyplot
from builtins import zip
import os
from scipy.optimize import curve_fit
from numpy import exp, power


# y= ax^3 + bx^2 + cx + d
def func(x, L,k,x0,y0):
    return  y0 + (L / (1.0 + exp( -k*(x-x0) )) )

if __name__ == '__main__':
    logdir = "compare_tree_model_logs"
#     logdir = "forgetting_logs"
    t_to_plot = ["RandomTutor"]
    m_to_plot = ["BranchMergeNetwork"]
#     m_to_plot = ["ConNTree","DivNTree"]
    b_to_plot = [2] #,3,4,5]
    #test_file = "itec2011"
    test_file =""
    fileext = ".log"
    

    pyplot.xlabel("# Steps")
    pyplot.ylabel("Mastery")
    
    files_to_plot = [t+m+str(b) for t in t_to_plot for m in m_to_plot for b in b_to_plot]
    print(files_to_plot)
    
    for t in files_to_plot:
        trace = []
        step_cnt = 0
        ep_len_cnt = 0
        filename = os.path.join(logdir, t+fileext);
        with open(filename) as f:
#             head = [next(f) for x in range(200)]
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
        
        paras, pcov = curve_fit(func, steps, score)
        fitted = [func(x, *paras) for x in steps]
        pyplot.plot(steps, fitted, label=t+" fitted")
        print(*paras)
        
        leg = pyplot.legend(loc='lower right')
        leg.get_frame().set_alpha(0.3)
    pyplot.show()