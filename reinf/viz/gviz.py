from _collections import defaultdict
from random import randint, uniform

import graphviz


def gviz_representation(pred_map):
    child_lookup =defaultdict(lambda: None)
    active_nodes = []
    for c in pred_map:
        if pred_map[c]:
            for p in pred_map[c]:
                if p not in child_lookup:
                    child_lookup[p]=[c]
                else:
                    child_lookup[p].append(c)

    rstr = "digraph {\n node [shape=\"circle\"];\n"

    for p in pred_map:
        c_list = child_lookup[p]
        if(c_list):
            c_str = " ".join(str(c.id) for c in c_list)
            rstr += str(p.id) + "->{" + c_str + "}\n"
    rstr+="}\n"
    return rstr



def gviz_representation(mod):
    child_lookup =defaultdict(lambda: None)
    active_nodes = []
    for c in mod.concepts:
        if c.predecessors:
            for p in c.predecessors:
                if p not in child_lookup:
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

def gvrender(pred_map, fname='g1'):
    g1 = graphviz.Digraph(format='png')
    for c in pred_map:
        g1.node(str(c))
        for entry in pred_map[c]:
            #p_id = "ROOT" if not p else p.id
            p_id = entry[0]
            wgt = entry[1]
            g1.edge(str(p_id), str(c), _attributes={"color":"#ff0000{:02X}".format(int(255*wgt))})
    print(g1.source)
    filename = g1.render(filename=fname)

if __name__=='__main__':
    pred_map = {}
    for c in range(10): # generate random graph
        pred_map[c] = set()
        for _ in range(randint(0,5)):
            rp = randint(0,10)
            wgt = uniform(0,1)
            pred_map[c].add((rp,wgt))
    gvrender(pred_map)