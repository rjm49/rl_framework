'''
Created on 11 Nov 2016

@author: Russell
'''
from reinf.exp1.classes import Concept
import matplotlib.pyplot as plt
from reinf.exp1.IdealLearner import IdealLearner
from reinf.exp1.domain_models import ChainDomain, FreeDomain, Div2Tree, Con2Tree, \
    BranchMergeNetwork
from matplotlib import legend
from reinf.exp1.tabular_tutor import SarsaTutor, RandomTutor
from _collections import defaultdict

def printNode(c):
    print(c.id)
    for p in c.predecessors:
        printNode(p)


N = 100  # number of nodes

batches = [
                {
                'run':True,
                'batch_name':'BMC3 - Random',
                'student': IdealLearner(),
                'model': BranchMergeNetwork(3),
                'tutor': "random",
#                 'Ks': [10*i for i in range(1,200)] + [1000*i for i in range(2,14)]
                'Ks': [i for i in range(1, 10000, 50)]
                },
             
                {
                'run':True,
                'batch_name':'BMC3 - Tabular',
                'student': IdealLearner(),
                'model': BranchMergeNetwork(3),
                'tutor': "tabular",
#                 'Ks': [10*i for i in range(1,200)] + [1000*i for i in range(2,14)]
                'Ks': [i for i in range(1, 10000, 50)]
                },

             ]


def learn_k_steps(K, tut, stu, dom):
    for k in range(K):
        ct = tut.choose_A(dom.concepts)
#                     print("concept picked=",ct.id)
        succ = stu.try_learn(ct)
        if succ:
#                         print("learned",ct.id,"successfully")
            tut.student_tried(ct, k)       
#                 print("".join(['X' if student.knows(n) else "-" for n in model.concepts]))
#                 print([(c.id,[p.id for p in c.predecessors]) for c in model.concepts])
    
    
    
if __name__ == '__main__':
    model = BranchMergeNetwork(3)
    model.regenerate(N)
            
    tutor = SarsaTutor()
#     tutor = RandomTutor()
    batch_msgs = []
    batch_x = []
    batch_y = []
    master_log = []
    name= "TabTut"
    
    student = IdealLearner()
       
    total_steps = 0
    score = 0.0
    k_steps = 100
    trials_per_model = 100
    
    ave_scores = defaultdict(lambda: 0.0)
    for i in range(trials_per_model):
        print("TRIAL",i)
        while score < 1.0:
            score = len(student.known_concepts) / len(model.concepts)
#             batch_x.append(total_steps)
#             batch_y.append(score)
            learn_k_steps(k_steps, tutor, student, model)
            
            print(total_steps, score)
            ave_scores[total_steps] += float(1/trials_per_model) * score
            total_steps += k_steps
            
    
    x, y = zip(*ave_scores.items())
    print(x,y)
    master_log.append((name, sorted(x), sorted(y), batch_msgs))

    #    print("Average score with ",K,"learning steps (across",trials_per_model,"trials =",ave_score)
    for s in master_log:
#         print(s[3])
        plt.plot(s[1], s[2], label=s[0])
    
    plt.ylabel('ave score (%s concepts, %s trials)' % (N, trials_per_model))
    plt.xlabel('learning iterations')
    leg = plt.legend(loc='right')
    leg.get_frame().set_alpha(0.3)
    plt.show()
