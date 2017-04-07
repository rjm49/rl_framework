'''
Created on 28 Mar 2017

@author: Russell
'''
from reinf.exp1.students.relearning import RelearningStudent
from reinf.exp1.classes import Concept
from matplotlib import pyplot

if __name__ == '__main__':
    
    s = RelearningStudent()
    c = Concept(0)
    
    ret_lvl = []
    tics = list(range(0,100))
    
    learn_at = [0, 30, 80]
    for tic in tics:
        if tic in learn_at:
            s.try_learn(c)
        else:
            s._decay_tick(1)
        r = s.known_concepts[c]
        ret_lvl.append(r)
        
    pyplot.xlabel("# Min")
    pyplot.ylabel("Retention")
    pyplot.ylim(0.0,1.0)

    pyplot.plot(tics, ret_lvl)
    pyplot.show()