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
    logdir = "compare_logs"
    traces_to_plot = ["RandomTutor", 
                      "Qutor",
                      "SarsaL2",
                      "DynaQutor"
                      ]
    #test_file = "itec2011"
    test_file =""
    fileext = ".log"
    

    pyplot.xlabel("# Steps")
    pyplot.ylabel("EpisodeLength")
    
    for t in traces_to_plot:
        trace = []
        step_cnt = 0
        ep_len_cnt = 0
        filename = os.path.join(logdir, t+test_file+fileext);
        with codecs.open(filename) as f:
            for line in f:
                #print(line.strip())
                #input("prompt")
                if(line.decode('UTF-8').strip()=='e') and ep_len_cnt:
                    trace.append((step_cnt, ep_len_cnt))
#                     print(trace)
                    ep_len_cnt=0
                else:
                    step_cnt+=1
                    ep_len_cnt+=1
                
    
        steps, score = zip(*trace)
        pyplot.plot(steps, score, label=t)
        
        leg = pyplot.legend(loc='upper right')
        leg.get_frame().set_alpha(0.3)
    pyplot.show()