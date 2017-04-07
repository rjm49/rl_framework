'''
Created on 7 Feb 2017

@author: Russell
'''
from _collections import defaultdict
import os

from reinf.exp1.domain_models import DivNTree, ConNTree, BranchMergeNetwork,\
    ChainDomain, FreeDomain
from reinf.exp1.students.ideal import IdealStudent
from reinf.exp1.tutors.random import RandomTutor
from reinf.viz.gviz import gvrender
from reinf.exp1.students.forgetting import ForgettingStudent
from reinf.exp1.students.relearning import RelearningStudent
from reinf.exp1.tutors.dynaqutor import DynaQutor
from reinf.exp1.tutors.dynaqutor2 import DynaQutor2
from reinf.exp1.policies.policy_utils import state_as_str


def main():

    n = 20
#     branchfs = [2,3,4,5]
    branchfs = [2]
    
#     log_dir = "..\\..\\compare_tree_model_logs\\"
#     log_dir ="..\\..\\forgetting_logs\\"
#     log_dir="..\\..\\bmc_only\\"
    log_dir ="..\\..\\dynaqutor_logs\\"
    
    write = True

    model_types=[BranchMergeNetwork]
#     model_types=[ ChainDomain, DivNTree,ConNTree,BranchMergeNetwork]
#     model_types = [ChainDomain]
    
    models =[]
    for m in model_types:
        for b in branchfs:
            mod = m(branch_factor=b)
            mod.regenerate(n)
            models.append(mod)
            print("created "+str(mod))
            gvrender(mod, os.path.join(log_dir, str(type(mod)).split(".")[-1][:-2]+str(b) ))
    
#     tut = RandomTutor(name="RandomTutor")
    tut = DynaQutor2(n, 0.55, 250, 1.0, "DynaQutor2")
    tut.MASTERY_THRESHOLD = 0.95
    tut.modelling_intensity = -1
    num_training_sessions=800
   
    scores =[]
    steps_to_teach=[]
    for m in models:
        if write: logfile = open(os.path.join(log_dir,tut.name.split(" ")[0]+ str(type(m)).split(".")[-1][:-2] + str(m.branch_factor)+".log"),"w")
        
#         m_scores = {}
#         m_scores[0] = 0
        tut.possible_actions = m.concepts
        s = tut.sRep.reset_state()
        tut.extend_Q(s, tut.possible_actions)
        
        step_cnt=0
        for i in range(num_training_sessions):
            p = RelearningStudent()
#             p = ForgettingStudent()
#             p= IdealStudent()
#             m.regenerate(n)
            tut.sRep.reset_state()
            mastery = 0.0
            ep_cnt =0
            while mastery < tut.MASTERY_THRESHOLD:
#                 tut.test_student(p)
                lesson, ex = tut.get_next_lesson()
                k = p.knows(lesson)
                success = p.try_learn(lesson)
                print(i,"/",step_cnt,"/",ep_cnt,str(tut.sRep),": tried to teach p",lesson.id,"with success=",success)
                mastery = p.get_mastery_score()/len(m.concepts)
                
                    
                tut.record_lesson_results(lesson, success, k, mastery, ex)
                
                step_cnt += 1
#                 m_scores[step_cnt]=mastery #/num_training_sessions
                scores.append(mastery)
                print("m=",mastery)
                ep_cnt+=1
            steps_to_teach.append([step_cnt, ep_cnt])
            pc=100.0*(i+1)/num_training_sessions
            if (pc == int(pc)):
                print("{}%".format(pc))

#         input("hit return")
        q = RelearningStudent()
#         q = IdealStudent()
        mastery = 0.0
        tut.sRep.reset_state()
        ep_cnt=0
        while mastery < tut.MASTERY_THRESHOLD:
#             tut.test_student(q)
            lesson, ex = tut.get_next_lesson()
            k= q.knows(lesson)
            success = q.try_learn(lesson)
            print(step_cnt, tut.sRep,": tried to teach q",lesson.id,"with success=",success)
            mastery = q.get_mastery_score()/len(m.concepts)
            tut.record_lesson_results(lesson, success, k, mastery, ex)
            step_cnt += 1
            ep_cnt +=1
            scores.append(mastery)
        steps_to_teach.append([step_cnt, ep_cnt])
#             m_scores[step_cnt]=mastery
#             print("m=",m_scores)
        
        print("len scores",len(scores))
#         for step_cnt in sorted(m_scores.keys()):            
#             mastery = m_scores[step_cnt]
        for step_cnt,mastery in enumerate(scores):
            if write:
                #print("writing log file")
                logfile.write("{},{}\n".format(step_cnt, mastery))

        if write: logfile.close()
 
        print(steps_to_teach)
 
if __name__ == '__main__':
    main()