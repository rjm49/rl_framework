'''
Created on 30 Dec 2016

@author: Russell
'''
from _collections import defaultdict
import graphviz
from reinf.exp1.domain_models import BranchMergeNetwork



def gviz_representation(mod):    
    child_lookup =defaultdict(lambda: None)
    active_nodes = []
    for c in mod.concepts:
        if c.predecessors:
            for p in c.predecessors:
                if p not in child_lookup.keys():
                    child_lookup[p]=[c]
                else:
                    child_lookup[p].append(c)

    rstr = "digraph {\n node [shape=\"circle\"];\n"

    for p in mod.concepts:
        c_list = child_lookup[p]
        if(c_list):
            c_str = " ".join(str(c.id) for c in c_list)
            rstr += str(p.id) + "->{" + c_str + "}\n"
    rstr+="}\n"
    return rstr

def gvrender(mod):
    g1 = graphviz.Digraph(format='png')
    for c in mod.concepts:
        g1.node(str(c.id))
        for p in c.predecessors:
            g1.edge(str(p.id), str(c.id))
    print(g1.source)
    filename = g1.render(filename='g1')

if __name__=='__main__':
    dom = BranchMergeNetwork(4)
    dom.regenerate(26)
    print(len(dom.concepts))
#     print(gviz_representation(dom))
    gvrender(dom)