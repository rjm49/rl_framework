import pickle

import numpy

from backfit.BackfitUtils import init_objects
from isaac.StudentSim import StudentSim
from isaac.QutorSim import Qutor
import random

BASE_DATA_DIR = "c:/git/IsaacData/"

# INGREDIENTS
# Simulated student
# RL Tutor
# Goal - first to 100 correct questions
target = 1000
n_users = 1000
# random.seed(666)
scores = []

cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs = init_objects(n_users)
predictor = pickle.load(open(BASE_DATA_DIR + "backfit/dw_cum_pred.pkl", "rb"))
print("loaded data")

all_qids = list(all_qids)
random.shuffle(all_qids)


actions = tuple(all_qids[0:100])
qutor = Qutor(alpha=0.667, eps=10, actions=actions)

# qutor.s = K
print("init'd Qutor")

print("starting loops...")
for x in range(5000):
    print("student {}".format(x))
    student = StudentSim(predictor)
    K = numpy.zeros(shape=len(cats))  # K33 vector encoding
    student.s = K
    Rtot = 0
    move_count = 0
    succ_count = 0
    fail_count = 0
    explorcnt = 0
    reps = 0
    for _ in range(100):  # what score can we get in 100 moves?
        A, explorative = qutor.choose_A()
        if explorative:
            explorcnt += 1
        # print(A, explorative)
        move_count += 1
        qenc = numpy.zeros(shape=len(cats))
        cat = cat_lookup[A]
        catix = cat_ixs[cat]
        # print(catix)
        diff = diffs[A]
        qenc[catix] = 1  # binary diff at this time

        # print("can",K,"pass",qenc,"?")
        if A in student.haveseen:
            reps += 1
        if student.doipass(A, qenc) == True:
            # print("pass")
            xK = numpy.copy(K)
            K[catix] = K[catix] + 1  # no learning rate specified yet!
            # print(K)
            # R = numpy.sqrt(numpy.sum(numpy.square(K))) - numpy.sqrt(numpy.sum(numpy.square(xK)))
            R = 1  # numpy.sum(K - xK)
            # R = numpy.linalg.norm(K - xK)
            # do qutor update here
            qutor.sa_update(xK, A, R, K)
            # print("success")
            # print("reward",R)
            succ_count += 1
        else:
            # print("F A I L")
            xK = numpy.copy(K)
            # K[catix] = 0  # no learning rate specified yet!
            # print(K)
            # R = numpy.sqrt(numpy.sum(numpy.square(K))) - numpy.sqrt(numpy.sum(numpy.square(xK)))
            R = 0  # numpy.sum(K - xK)
            # R = numpy.linalg.norm(K-xK)
            # do qutor update here
            qutor.sa_update(xK, A, R, K)
            # print("failure")
            # print("reward",R)
            fail_count += 1
        Rtot += R
    # print("end K", K)
    # print("student #{}, for {} moves, succ moves {} v fails {}, total score {}".format(x, move_count, succ_count,
    #                                                                                    fail_count, Rtot))
    # print("exc", explorcnt)
    # print("repetitions", reps)
    sc, ac = qutor.status_report()
    Ks = "[" + " ".join(map(str,K)) + "]"
    print("final", Ks, "score=", numpy.sum(K))
    scores.append((x, Ks, move_count, succ_count, explorcnt, reps, sc, ac))

# input("prompt")
f = open("qutor.csv", "w")
for t in scores:
    f.write(",".join(map(str, t)) + "\n")
f.close()
