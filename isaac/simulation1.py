import pickle

import numpy
import pandas

from backfit.BackfitUtils import init_objects
from backfit.utils.utils import load_new_diffs, load_mcmc_diffs
from isaac import itemencoding
from isaac.QutorGen import gen_X_primed, gen_qenc
from isaac.QutorSimFA import QutorFA
from isaac.StudentSim import StudentSim
from isaac.QutorSim import Qutor
import random

from matplotlib import pyplot as plt

# INGREDIENTS
# Simulated student
# RL Tutor
# Goal - first to 100 correct questions
target = 1000
n_users = 1000
# random.seed(666)
scores = []

cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs = init_objects(n_users, path="../../isaacdata/")
passrates, stretches, passquals, all_qids = load_new_diffs("../../isaacdata/pass_diffs.csv")
mcmcdiffs = load_mcmc_diffs("../../isaacdata/mcmc/mcmc_results.csv")
predictor = pickle.load(open("qutor_test.pkl", "rb"))
scaler = pickle.load(open("qutor_scaler.pkl", "rb"))
print("loaded data")

all_qids = list(all_qids)
random.shuffle(all_qids)

actions = tuple(all_qids)
# qutor = Qutor(alpha=0.1, gamma=1.0, eps=1000, actions=actions)
qutor = QutorFA(0.1, 1.0, 1000, actions, "QutorFA", cat_lookup, cat_ixs, passrates, stretches, levels, passquals)

# qutor.s = K
print("init'd Qutor")

print("starting loops...")
for x in range(50000):
    print("student {}".format(x))
    student = StudentSim(predictor, scaler)
    K = numpy.zeros(shape=(itemencoding.n_components, itemencoding.k_features))  # K33 vector encoding
    student.s = K
    Rtot = 0
    move_count = 0
    succ_count = 0
    fail_count = 0
    explorcnt = 0
    reps = 0

    alpha=1.0
    phi= 0.5
    passed = False

    for _ in range(100):  # what score can we get in 100 moves?
        A, explorative = qutor.choose_A(K)
        if explorative:
            explorcnt += 1
        # print(A, explorative)
        move_count += 1
        catix = cat_ixs[cat_lookup[A]]

        qenc = gen_qenc(catix, passrates[A], stretches[A], levels[A], passquals[A])

        if A in student.haveseen:
            reps += 1

        if student.doipass(A, K, qenc) == True:
            R= 1.0
            succ_count += 1
            passed = True
        else:
            R = -1.0
            fail_count += 1
            passed = False

        #print(K)
        xK = numpy.copy(K)
        K = gen_X_primed(K, catix, alpha, phi, passed, passrates[A], passquals[A], stretches[A])
        #print("PRIMD")
        #print(K)
        qutor.sa_update(xK, A, R, K)
        Rtot += R
        move_count += 1
    sc, ac = qutor.status_report()
    Ks = "[" + " ".join(map(str,K)) + "]"
    print("final", Ks, "score=", numpy.sum(K))
    scores.append((x, succ_count))

df = pandas.DataFrame.from_records(scores, columns=["trial","successes"])
df.to_csv("qutor.csv")

print("plotting")
df[0::100].plot(x='trial', y=['successes'])
plt.show()