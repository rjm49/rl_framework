'''
Created on 1 Feb 2017

@author: Russell
'''
from reinf.viz.gviz import gviz_representation, gvrender
import codecs
from reinf.exp1.domain_models import BranchMergeNetwork, Domain
import re
from reinf.exp1.classes import Concept
from reinf.exp1.policies.policy_utils import state_as_str
import copy

def load_concepts_from_file(fname):
    concepts = {}
    f = codecs.open(fname, "r")
    lines = f.readlines()
    print("loading concepts")
    print(lines)
    
    #get simple metadata
    name = lines.pop(0)
    klass = lines.pop(0)
    num_nodes = int(lines.pop(0))
    branchfactor = int(lines.pop(0))
    
    for line in lines:
        line = line.strip()
        if "->" in line:
            parts = re.sub(" ", "", line).split("->")
            pred = parts[0]
            chdn = re.sub("[{}]", "", parts[1]).split(" ")

            ip = int(pred)
            if ip in concepts:
                p=concepts[ip]
            else:
                p=Concept(ip)
                concepts[ip]=p
            
            for c in chdn:
                ix = int(c)
                if ix in concepts:
                    ct = concepts[ix]
                else:
                    ct = Concept(ix)
                    concepts[ix]=ct
                ct.predecessors.append(p)
#     return list(c.id for c in concepts.values())
    return name, klass, num_nodes, branchfactor, list(concepts.values())

def save_domain_to_file(d, fname):
    f = codecs.open(fname, "w")
    f.write(d.name+"\n")
    f.write(d.__class__.__name__+"\n")
    f.write(str(len(d.concepts))+"\n")
    f.write(str(d.branch_factor)+"\n")
    s = gviz_representation(d)
    f.write(s)
    f.close()
    
    
def score_domain_similarity(dref, dinf):
    err = 0.0
    arcs = 0.0
    for cref in dref.concepts:
        cinf= dinf.concepts[cref.id]
        arcs += len(cref.predecessors)
        for pguess in cinf.predecessors:
            if pguess.id not in [c.id for c in cref.predecessors]:
                err += 1.0
                
    print(arcs,err,1+err/arcs)
    #input("prompt")
    return 1.0+(err/arcs)

def load_domain_model_from_file(fname):
    name, klassname, num_nodes, branchfactor, concepts = load_concepts_from_file(fname)
    klass = eval(klassname)
    if branchfactor: #TODO other domain models need branchfactor support (0 for "no branching")
        dom2 = klass(branchfactor)
    else:
        dom2 = klass()
    dom2.name = name
    dom2.regenerate(num_nodes) 
    dom2.concepts = concepts
    return dom2

if __name__=="__main__":
    dom = BranchMergeNetwork(2)
    dom.regenerate(10)
    save_domain_to_file(dom, "f.dat")
    name, klassname, num_nodes, branchfactor, concepts = load_concepts_from_file("f.dat")
    
    klass = eval(klassname)
    dom2 = klass(branchfactor)
#    dom2 = BranchMergeNetwork(2)

    dom2.name = name
    dom2.regenerate(num_nodes) 
    dom2.concepts = concepts
    
    print(dom2)