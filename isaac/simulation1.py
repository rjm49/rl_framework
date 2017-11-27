import pickle

import numpy
import pandas

from backfit.BackfitUtils import init_objects
from backfit.utils.utils import load_new_diffs, load_mcmc_diffs
from isaac import itemencoding
from isaac.QutorGen import gen_X_primed, gen_qenc
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

cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs = init_objects(n_users)
passrates, stretches, passquals, all_qids = load_new_diffs("../../isaacdata/pass_diffs.csv")
mcmcdiffs = load_mcmc_diffs("../../isaacdata/mcmc/mcmc_results.csv")
predictor = pickle.load(open("qutor_test.pkl", "rb"))
print("loaded data")

all_qids = list(all_qids)
random.shuffle(all_qids)
all_qids = all_qids[0:20]

actions = tuple(all_qids)
qutor = Qutor(alpha=0.1, gamma=1.0, eps=100, actions=actions)

# qutor.s = K
print("init'd Qutor")

print("starting loops...")
for x in range(128):
    print("student {}".format(x))
    student = StudentSim(predictor)
    K = numpy.zeros(shape=(itemencoding.n_components, itemencoding.k_features))  # K33 vector encoding
    student.s = K
    Rtot = 0
    move_count = 0
    succ_count = 0
    fail_count = 0
    explorcnt = 0
    reps = 0

    alpha=1.0
    phi= 1.0

    for _ in range(20):  # what score can we get in 100 moves?

        A, explorative = qutor.choose_A(K)
        if explorative:
            explorcnt += 1
        # print(A, explorative)
        move_count += 1

        catix = cat_ixs[cat_lookup[A]]

        qenc = gen_qenc(catix, passrates[A], stretches[A], levels[A], passquals[A])
        print(K.flatten().shape)
        print(qenc.flatten().shape)

        # print("can",K,"pass",qenc,"?")
        if A in student.haveseen:
            reps += 1
        if student.doipass(A, K, qenc) == True:
            # print("pass")
            xK = numpy.copy(K)
            #K[catix] = K[catix] + 1  # no learning rate specified yet!
            K = gen_X_primed(K, catix, alpha, phi, True, passrates[A], passquals[A], stretches[A])
            #R = numpy.sum(numpy.square(K)) - numpy.sum(numpy.square(xK))
            #R = 1.0
            #R = numpy.sqrt(K.dot(K)) - numpy.sqrt(xK.dot(xK))
            R= 1.0
            qutor.sa_update(xK, A, R, K)
            succ_count += 1
        else:
            # print("F A I L")
            xK = numpy.copy(K)
            K = gen_X_primed(K, catix, alpha, phi, False, passrates[A], passquals[A], stretches[A])
            # K[catix] = 0  # no learning rate specified yet!
            # print(K)
            R = -1.0
            # R = numpy.linalg.norm(K-xK)
            qutor.sa_update(xK, A, R, K)
            fail_count += 1
        Rtot += R
        move_count += 1
    # print("end K", K)
    # print("student #{}, for {} moves, succ moves {} v fails {}, total score {}".format(x, move_count, succ_count,
    #                                                                       fail_count, Rtot))
    # print("exc", explorcnt)
    # print("repetitions", reps)
    sc, ac = qutor.status_report()
    Ks = "[" + " ".join(map(str,K)) + "]"
    print("final", Ks, "score=", numpy.sum(K))
    scores.append((x, Ks, move_count, numpy.sum(K), Rtot, explorcnt, reps, sc, ac))

df = pandas.DataFrame.from_records(scores, columns=["trial","endK","moves","successes", "Rtot","explorn","repeats","#states","#actions"])
df.to_csv("qutor.csv")

print("plotting")
df[0::500].plot(x='trial', y=['successes'])
plt.show()