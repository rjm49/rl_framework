'''
Created on 14 Nov 2016

@author: Russell
'''
from _collections import defaultdict
from abc import abstractmethod
from math import sqrt
import random

from reinf.exp1.classes import Concept
from reinf.viz.gviz import gvrender

class Domain(object):
    '''
    This is an abstract class to hold place for various Knowledge Domain objects.
    These objects have a corpus of Concepts that are related by an underlying necessity/sufficiency structure.
    '''
    def __init__(self, branch_factor=0):
        self.concepts = []
        self.name = "Un-named"
        self.branch_factor= branch_factor
        
    @abstractmethod
    def regenerate(self, n, branch_factor=2):
        self.concepts = []

class ChainDomain(Domain):
    '''This Domain model has Concepts that are linked together in a chain structure, i.e. a linked list.'''
    
    def regenerate(self, n, branch_factor=0):
        Domain.regenerate(self, n)
        last_c = None    
        for c_id in range(n):
            c = Concept(c_id)
            self.concepts.append(c)
            if last_c:
                c.predecessors.append(last_c)
            last_c = c
            
class FreeDomain(Domain):
    '''This Domain model has total independent Concepts that can be approached in any order'''    
    def regenerate(self, n, branch_factor=0):
        Domain.regenerate(self, n)
        for c_id in range(n):
            c = Concept(c_id)
            self.concepts.append(c)
            c.predecessors=[]
            
class DivNTree(Domain):
    def regenerate(self, n):
        Domain.regenerate(self, n, branch_factor=0)
        for c_id in range(n):
            c = Concept(c_id)
            self.concepts.append(c)
        
        c_pnt=0        
        for p in self.concepts:
            for i in range(self.branch_factor):
                c_pnt+=1
                if c_pnt >= len(self.concepts):
                    break
                else:
                    ch = self.concepts[c_pnt]
                    ch.predecessors.append(p)
                      

class ConNTree(Domain):    
    def regenerate(self, n, branch_factor=0):
        Domain.regenerate(self, n)
        for c_id in range(n):
            c = Concept(c_id)
            self.concepts.append(c)
            
        p_pnt=0
        rv_concepts = self.concepts[::-1]
        for c in self.concepts:
            for i in range(self.branch_factor):
                p_pnt+=1
                if p_pnt >= len(self.concepts):
                    break
                else:
                    pt = self.concepts[p_pnt]
                    c.predecessors.append(pt)


class RandomNetwork(Domain):
    def regenerate(self, n, branch_factor=0):
        Domain.regenerate(self, n)
        
        childfree = []
        
        for c_id in range(n):
            c = Concept(c_id)
            self.concepts.append(c)
            childfree.append(c)
                
        while(len(childfree)>1):
            child_cand = random.choice(self.concepts)
            par_cand = random.choice(self.concepts)
#             print(par_cand.id, " ->? ", child_cand.id)
            if(par_cand!=child_cand) and not par_cand.has_ancestor(child_cand) and not (child_cand.has_ancestor(par_cand)):
                child_cand.predecessors.append(par_cand)
#                 print(par_cand.id, "->", child_cand.id)
                if(par_cand in childfree): childfree.remove(par_cand)
        
class BranchMergeNetwork(Domain):
    
    def __init__(self, branch_factor=2):
        self.branch_factor = branch_factor
        self.name = 'BranchMergeNetwork'
    
    def regenerate(self, n):
        Domain.regenerate(self, n)

        deck = []
        for c_id in range(n):
            c = Concept(c_id)
            self.concepts.append(c)
            deck.append(c)
                   
        #random.shuffle(deck) # shuffle the concepts into random order
        
        #n_entry = random.randint(1,int(sqrt(len(deck))))
        #n_entry = int(sqrt(len(deck)))
        n_entry=1
        active_nodes = [deck.pop(0) for _ in range(n_entry)]
        
#         print(n_entry,"entry nodes", [e.id for e in active_nodes])
        
        while(deck):
            op = random.randint(0,1)
#             print([n.id for n in active_nodes],[d.id for d in deck], op)
            if(op==0): #merge
                if(len(active_nodes)>=2) and deck:
                    ps=[]
                    for _ in range(self.branch_factor):
                        if active_nodes:
                            ps.append( active_nodes.pop(0) )
                    
                    child = deck.pop(0)
                    for p in ps:
                        child.predecessors.append(p)
                    active_nodes.append(child)

            elif(op==1): #split
                if len(deck)>=2:
                    p1 = active_nodes.pop(0)
                    cs = []
                    for _ in range(self.branch_factor):
                        if deck:
                            c = deck.pop(0)
                            c.predecessors.append(p1)
                            active_nodes.append(c)
            else: #crain
                if deck:
                    p = active_nodes.pop(0)
                    c = deck.pop(0)
                    c.predecessors.append(p)
                    active_nodes.append(c)

def tikz_representation(mod):    
    child_lookup =defaultdict(lambda: None)
    active_nodes = []
    for c in mod.concepts:
        if c.predecessors:
            for p in c.predecessors:
                if p not in child_lookup.keys():
                    child_lookup[p]=[c]
                else:
                    child_lookup[p].append(c)
        else:
            active_nodes.append(c)

    for i,f in enumerate(active_nodes):
        if i==0:
            print("\\node[state] (%d) {%d};" % (f.id, f.id))
        else:
            print("\\node[state] (%d) [right=2cm of %d] {%d};" % (f.id, active_nodes[i-1].id, f.id ))
    
    while(active_nodes):
        f = active_nodes.pop(0)
        c_list = child_lookup[f]
        
        #print(f.id, "-lookup->", [c.id for c in c_list])
        
        if c_list:
            if len( c_list )==1:
                c = c_list[0]
                if(len(c.predecessors)==1):
                    print("\\node[state] (%d) [below of=%d] {%d};" % (c.id, c.predecessors[0].id, c.id))
                else:
                    print("\\node[state] (%d) [below right of=%d] {%d};" % (c.id, c.predecessors[0].id, c.id))
                    active_nodes.pop(0) #discard the second predecessor
            else:
                c1_id = c_list[0].id
                c2_id = c_list[1].id
                print("\\node[state] (%d) [below of=%d] {%d};" % (c1_id, f.id, c1_id))
                print("\\node[state] (%d) [below right of=%d] {%d};" % (c2_id, f.id, c2_id))
            active_nodes += c_list #extend the active nodes list with the children
#             print([a.id for a in active_nodes])
    
    s = "\\path "
    for c in mod.concepts:
        c_list = child_lookup[c]
        if c_list:
            for cc in c_list:
                s+="(%d) edge node {} (%d)\n" % (c.id, cc.id)
    s+=";"
    print(s)
    
if __name__=="__main__":
    d = ConNTree(branch_factor=3)
    d.regenerate(10)
    gvrender(d, fname="c")