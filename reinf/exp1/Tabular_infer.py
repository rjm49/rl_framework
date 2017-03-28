'''
Created on 11 Nov 2016

@author: Russell
'''
import codecs

from matplotlib import pyplot as plt

from reinf.exp1.domains.domain_utils import load_domain_model_from_file
from reinf.exp1.students.ideal import IdealStudent
from reinf.exp1.tutors.random import RandomTutor
from reinf.exp1.domains.filterlist_utils import clean_filterlist
from reinf.exp1.policies.policy_utils import state_as_str
from reinf.exp1.classes import Concept
from email._header_value_parser import Domain
from reinf.viz.gviz import gvrender
import copy
from decimal import DivisionByZero
from reinf.exp1.tutors.qutor import Qutor
from reinf.exp1.tutors.sarsa_lambda_2 import SarsaL2
from _collections import defaultdict
import graphviz
from test.test_inspect import attrs_wo_objs
from reinf.exp1.tutors.dynaqutor import DynaQutor


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
    
    fname = "itec2011"
    model = load_domain_model_from_file(fname+".dat")    
    gvrender(model, fname)
    train_logs = []
    batch_names =[]
    
    num_nodes=len(model.concepts)
#     tutor = RandomTutor(num_nodes=num_nodes)
#     tutor = Qutor(num_nodes, 0.5, 100, 1.0, "Qutor") #0.1 7000
#     tutor = Qutor(num_nodes, 0.1, 7000, 1.0, "Qutor") #0.1 7000
#     tutor = SarsaL2(num_nodes, 0.5, 5000, 1.0, "SarsaL2")
    tutor = DynaQutor(num_nodes, 0.5, 1000, 1.0, "DynaQ")
    
    for _ in range(100):
        tutor.reset()
        p = IdealStudent()
        tutor.run_episode(model, p, -1, True)
    
    trace = tutor.transition_trace
    
    fl = {}
    for c in model.concepts:
        fl[c.id]=[True]*num_nodes
        
    ps=[]
    rs=[]
    Fs=[]
    clns=[]

    steps=[] #keep a list of all steps made across all episodes    
    for episode in trace: # join all the separate episodes together
        steps += episode

    f = codecs.open(str(tutor)+"_trace_on_"+fname,"w")
    for s in steps:
        f.write(str(s)+"\n")
    f.close()

    t_to_a = defaultdict(int)
    t_pred_cnt = defaultdict(lambda: defaultdict(int))
    cond_probs = defaultdict(lambda: defaultdict(int))
    
    cnt_of_kc = defaultdict(int)
    cnt_a_attempts_kc= defaultdict(lambda: defaultdict(int))
    
    entries = set()
    leaves = set()

    cnt_a_attempts=defaultdict(int)
    cnt_a_successs = defaultdict(int)
    cnt_a_successs_kc = defaultdict(lambda: defaultdict(int))
    
    for step in steps:
        s_str,a_id,succ = step
        s_blns = [bool(int(x)) for x in s_str]

#         for ix,sb in enumerate(s_blns):
#             if sb:
#                 cnt_of_kc[ix]+=1
#         cnt_a_attempts[a_id]+=1
        for ix,sb in enumerate(s_blns):
            if sb and not s_blns[a_id]:
                cnt_a_attempts_kc[a_id][ix]+=1
                if succ:
                    cnt_a_successs_kc[a_id][ix] += 1
#             cnt_a_successs[a_id]+=1

        if succ:
#             print("SBLNS",s_blns)
            if True not in s_blns:
                entries.add(a_id)
            if s_blns.count(False)==1:
                leaves.add(a_id)

        if succ:
            record = fl[a_id] # lazy initialise
            new = [(s and r) for s,r in zip(s_blns,record)]
            print(a_id, state_as_str(s_blns),state_as_str(record),"->",state_as_str(new))
            fl[a_id]=new

            t_to_a[a_id]+=1 
            #here we want the cond probs P(knows(S) | A)
            sids = [ i for i,s in enumerate(s_blns) if s ]
            for sid in sids:
#                 print(a_id, sid)
#                 print(t_pred_cnt[a_id])
                t_pred_cnt[a_id][sid] =  t_pred_cnt[a_id][sid] +1


        cln = clean_filterlist(fl) #remove redundant arcs
        p,r,F=_score_similarity(model, cln)
        ps.append(p)
        rs.append(r)
        Fs.append(F)

    print(entries)
    print(leaves)
#     input("prompt")

    
    g1w = graphviz.Digraph(format='png', name="all_condprobs")
    g1 = graphviz.Digraph(format='png', name="deps_condprob")
    g2 = graphviz.Digraph(format='png', name="dependencies")
    gagc = graphviz.Digraph(format='png', name="condprobs_gagc")
#     for c in model.concepts:
#         #g1.node(str(c.id))
#         a = c.id
#         if a in t_to_a:
#             t = t_to_a[a]
#             for s in t_pred_cnt[a]:
#                 pred_c = t_pred_cnt[a_id][s]
#                 cond_prob=pred_c/float(t)
#                 print("P(knows({})|{})={}".format(s,a,cond_prob))
    
    E = set() # set of graph edges
    V = set() # vertex set
    PE = set()
    cp = {}

    p_of_a_given_c = defaultdict()    
    edges = {}
    for a in sorted(cnt_a_attempts_kc):
        for c in sorted(cnt_a_attempts_kc[a]):
            att_kc = cnt_a_attempts_kc[a][c] # how many times you attempt A knowing C
            suc_kc = cnt_a_successs_kc[a][c] #how many times you successfully learn A knowing C
            print(a,c)
            print(att_kc,suc_kc)

            p = suc_kc/att_kc
            if (a,c) in edges and edges[(a,c)][0]==p:
                edges[(a,c)]=(p,True) #set edge to be bidirectional
            else:
                edges[(c,a)]=(p,False)
            
            print("p({} | k({}))={}".format(a, c, suc_kc/att_kc))
    
    for e in edges:
        e0,e1 = e
        p,bdir = edges[e]
        alp = hex(int(p*255))[2:]    
        col='#000000' if p==1.0 else '#660000'
        if bdir:
            gagc.edge(str(e0), str(e1), _attributes={'color':col, 'dir':'both'})
        else:
            if p>=0.4:
                gagc.edge(str(e0), str(e1), _attributes={'color':col+alp})

    for k in p_of_a_given_c:
        p = p_of_a_given_c[k]
#         print("p({}|{})={}".format(k[0],k[1], p))
    for a in t_to_a:
        t = t_to_a[a]
        for s in t_pred_cnt[a]:
            pred_c = t_pred_cnt[a][s]
            cond_prob=pred_c/float(t)
#             print("P(knows({})|{})={}".format(s,a,cond_prob))
            
            alp = hex(int(cond_prob*255))[2:]
            g1w.edge(str(s), str(a), _attributes={'color':'#000000'+alp})
            cp[(s,a)]=cond_prob

            if cond_prob == 1.0:
                E.add((s,a))
                V.add(s)
                V.add(a)
            else:
                PE.add((s,a, cond_prob))

    E_ = copy.copy(E)
    for v in V:
        xn = [e for e in E_ if e[0]==v]
        print(xn)
        #input("prompt xn")
        for e in xn:
            x = e[1]
            gxn = [j for j in E_ if j[0]==x]
            print(gxn)
            #input("prompt gxn")
            for g in gxn:
                gd= g[1]
                if (v,gd) in E_:
                    print("removing",(v,gd))
                    E_.remove((v,gd))
    
    print(E_)
    
    for e in E_:
        g1.edge(str(e[0]), str(e[1]))
        g2.edge(str(e[0]), str(e[1]))
    
    for e in PE:
        alp = hex(int(e[2]*255))[2:]
        if True or e[2]>0.5:
            g1.edge(str(e[0]), str(e[1]), _attributes={'color':'#005500'+alp})
    
#     print(g1.source)
    #g1.render(filename="vizzy")
#     g1w.view()
    g1.view()
    g2.view()
    gagc.view()

    fl = clean_filterlist(fl)
#     for a_id in sorted(fl):
#         print(a_id, state_as_str(fl[a_id]))

#     input("hit key")

    dummod = Domain()
    dummod.concepts = [Concept(i) for i in range(num_nodes) ]
    for k,v in fl.items():
        print(k, state_as_str(v))
        con = dummod.concepts[k]
        pixs = [ix for ix,bl in enumerate(v) if bl ] #get ids where state entry is True
        for i in pixs:
            con.predecessors.append(dummod.concepts[i])            
    gvrender(dummod, "inferred_from_trace")
   

    plt.ylabel("Dependency detection")
    plt.xlabel('# Steps')

    step_axis = [ (i+1) for i,v in enumerate(steps)]

#     plt.plot( step_axis, ps, label="Precision")
#     plt.plot( step_axis, rs, label="R")
    plt.plot( step_axis, Fs, label=str(tutor)+"_F1")
    leg = plt.legend(loc='right')
    leg.get_frame().set_alpha(0.3)
    plt.ylim((0.0,1.0))
    plt.show()
        
