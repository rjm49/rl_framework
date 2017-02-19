'''
Created on 8 Feb 2017

@author: Russell
'''
from email._header_value_parser import Domain
from reinf.exp1.classes import Concept
from reinf.exp1.policies.policy_utils import state_as_str
from reinf.viz.gviz import gvrender
import copy


def update_filter(filterlist, old_S, A_id, succ):
    if succ: # we moved from old_S -A-> new_S
        #one or more of the bits from old_S enabled A to happen
        entry = filterlist[A_id]
        new_S = [a and b for a,b in zip(old_S, entry)]
        filterlist[A_id] = new_S
        print("\nfilterlist",A_id,"was ",state_as_str(entry),"^",state_as_str(old_S),"=>", state_as_str(new_S))
        
def _get_idxs(blist):
    return [i for i,x in enumerate(blist) if x==True]

#PRUNING
#If I share a predecessor with any of my predecessors, delete mine
def clean_filterlist(filterlist):
    cleanlist=copy.copy(filterlist)
#     print("CLEAN FL")
    for my_id in filterlist:            
        my_preds = _get_idxs(filterlist[my_id])
#         print("my_id",my_id,"my_preds",my_preds)
        for p_id in my_preds:
            grandparents = _get_idxs(filterlist[p_id])
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