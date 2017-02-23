'''
Created on 11 Nov 2016

@author: Russell
'''
import codecs

from matplotlib import pyplot as plt

from reinf.exp1.domains.domain_utils import load_domain_model_from_file
from reinf.exp1.students.ideal import IdealLearner
from reinf.exp1.tutors.random import RandomTutor
from reinf.exp1.domains.filterlist_utils import clean_filterlist
from reinf.exp1.policies.policy_utils import state_as_str
from reinf.exp1.classes import Concept
from email._header_value_parser import Domain
from reinf.viz.gviz import gvrender
import copy
from decimal import DivisionByZero
from reinf.exp1.tutors.qutor_goalbased import Qutor


def _score_similarity(dref, inf_fl):
    err = 0
    fp=0
    tp=0
    fn=0
    tn=0
    arcs=0
    for cref in dref.concepts:
        cinf= inf_fl[cref.id]
        arcs += len(cref.predecessors)
        real_prids = [c.id for c in cref.predecessors]
        for i,pguess in enumerate(cinf):
            if pguess:
                if (i in real_prids):
                    tp+=1
                else:
                    fp+=1
            else:
                if (i not in real_prids):
                    tn+=1
                else:
                    fn+=1
                
    try:
        p = tp / float(tp+fp)
    except ZeroDivisionError:
        p = 1.0

    try:
        r = tp / float(tp+fn)
    except DivisionByZero:
        r= 1.0
#   r = tp/ float(arcs)
#     print(p,r)

    F = 0.0 if (p+r==0) else (2.0*p*r / (p+r))


    return p,r,F

if __name__ == '__main__':
    #create some chain of Concepts
    
    #create a single model upon which all our efforts will focus
    
    
    fname = "itec2011"
    model = load_domain_model_from_file(fname+".dat")    
    gvrender(model, fname)
    train_logs = []
    batch_names =[]
    
#     tutor = RandomTutor(num_nodes=len(model.concepts))
    tutor = Qutor(len(model.concepts), 0.1, 5000, 1.0, "Qutor")
   
    for _ in range(200):
        tutor.reset()
        p = IdealLearner()
        tutor.run_episode(model, p, -1, False)
    
    trace = tutor.transition_trace
    n = tutor.num_nodes
    fl = {}
    for c in model.concepts:
        fl[c.id]=[True]*n
        
    ps=[]
    rs=[]
    Fs=[]
    clns=[]

    steps=[] #keep a list of all steps made across all episodes    
    for e in trace:
        steps += e
    
    for step in steps:
        s_str,a_id,succ = step
        if succ:
            s_blns = [bool(int(x)) for x in s_str]
            record = fl[a_id] # lazy initialise
            new = [(s and r) for s,r in zip(s_blns,record)]            
            print(a_id, state_as_str(s_blns),state_as_str(record),"->",state_as_str(new))
            fl[a_id]=new

#             cln=fl
        cln = clean_filterlist(fl)
#         clns.append(cln)
        p,r,F=_score_similarity(model, cln)
        ps.append(p)
        rs.append(r)
        Fs.append(F)
    
#     for c in clns:
#         print(c)
    
    
    fl = clean_filterlist(fl)
#     for a_id in sorted(fl):
#         print(a_id, state_as_str(fl[a_id]))

#     input("hit key")

    dummod = Domain()
    dummod.concepts = [Concept(i) for i in range(n) ]
    for k,v in fl.items():
        print(k, state_as_str(v))
        con = dummod.concepts[k]
        pixs = [ix for ix,bl in enumerate(v) if bl ] #get ids where state entry is True
        for i in pixs:
            con.predecessors.append(dummod.concepts[i])            
    gvrender(dummod, "inferred_from_trace")
   

        #         print(s[3])
    plt.ylabel("Dependency detection")
    plt.xlabel('# Steps')

    step_axis = [ (i+1) for i,v in enumerate(steps)]

    plt.plot( step_axis, ps, label="Precision")
    plt.plot( step_axis, rs, label="Recall")
    plt.plot( step_axis, Fs, label="F1")
    leg = plt.legend(loc='right')
    leg.get_frame().set_alpha(0.3)
    plt.show()
        
