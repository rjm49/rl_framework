'''
Created on 8 Feb 2017

@author: Russell
'''
from email._header_value_parser import Domain
from reinf.exp1.classes import Concept
from reinf.exp1.policies.policy_utils import state_as_str
from reinf.viz.gviz import gvrender
import copy
from _collections import defaultdict


def update_filter(filterlist, old_S, A_id, succ):
    if succ: # we moved from old_S -A-> new_S
        #one or more of the bits from old_S enabled A to happen
        entry = filterlist[A_id]
        new_S = [a and b for a,b in zip(old_S, entry)]
        filterlist[A_id] = new_S
        #print("\nfilterlist",A_id,"was ",state_as_str(entry),"^",state_as_str(old_S),"=>", state_as_str(new_S))
        
def _get_idxs(blist):
    return [i for i,x in enumerate(blist) if x==True]

#PRUNING
#If I share a predecessor with any of my predecessors, delete mine
def clean_filterlist(filterlist):
    cleanlist={}
    for k in filterlist:
        cleanlist[k]=copy.copy(filterlist[k])
        
#     print("CLEAN FL")
    for my_id in cleanlist:            
        my_preds = _get_idxs(cleanlist[my_id])
#         print("my_id",my_id,"my_preds",my_preds)
        for p_id in my_preds:
            grandparents = _get_idxs(cleanlist[p_id])
#             print("p_id",p_id,"grandps",grandparents)
            for grandparent_id in grandparents:
                if grandparent_id in my_preds:
#                     print("BEFORE", my_id, state_as_str(cleanlist[my_id]))
                    cleanlist[my_id][grandparent_id]=False
#                     print("AFTER:", my_id, state_as_str(cleanlist[my_id]))
#                     input("prompt")
    return cleanlist


def build_inferred_model(mod, tutor, after_k_interations):
    tutor.filterlist = clean_filterlist(tutor.filterlist)
    dummod = Domain()
    dummod.concepts = [Concept(i) for i,c in enumerate(mod.concepts) ]
    for k,v in tutor.filterlist.items():
        print(k, state_as_str(v))
        con = dummod.concepts[k]
        pixs = [ix for ix,bl in enumerate(v) if bl ] #get ids where state entry is True
        for i in pixs:
            con.predecessors.append(dummod.concepts[i])            
    gvrender(dummod, "inferred"+str(after_k_interations))
    return dummod

def print_success_history_totals(sh):
    scores = defaultdict(lambda: defaultdict(int))
    for ep in sh:
        for step in ep:
            a_id = step[1]
            s_str = step[0]
            passed = step[2]
            for i,ch in enumerate(s_str):
                inc = int(ch) # convert to 1 or 0
                scores[a_id][i]+= inc if passed else 0

    actions = sorted([a for a in scores])
    for a in actions:
        scs = scores[a]
        scs_list = [ str(p)+":"+str(scs[p]) for p in sorted(scs.keys()) ]
        print(a, scs_list)
        
def intersect_all_history_totals(sh):
    pred_cands = {}    
    for ep in sh:
        for step in ep:
            a_id = step[1]
            s_str = step[0]
            passed = step[2]
    
            if passed:
                if a_id not in pred_cands:
                    pred_cands[a_id]=s_str #the first success state is our initial candidate for A's predecessors 
#                     print("adding",a_id,"=",s_str)
                else:
                    entry = pred_cands[a_id]
#                     print("entry at",a_id,"=",entry)
                    new_str = "".join([str(int(int(a) and int(b))) for a,b in zip(s_str, entry)])
                    pred_cands[a_id]=new_str
            
    for a in pred_cands:
        print(a, pred_cands[a])

        
    ks = sorted(pred_cands.keys())
    
    dummod = Domain()
    dummod.concepts = [Concept(i) for i,c in enumerate(ks) ]
    for a_id,s_str in pred_cands.items():
        print(a_id, s_str)
        con = dummod.concepts[a_id]
        pixs = [ix for ix,bl in enumerate(s_str) if bl=="1" ] #get ids where state entry is True
        for i in pixs:
            con.predecessors.append(dummod.concepts[i])
#     gvrender(dummod, "inferred")
    return dummod