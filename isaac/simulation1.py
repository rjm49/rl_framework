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
from isaac.itemencoding import create_S

target = 1000
n_users = 1000
# random.seed(666)
scores = []

cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs = init_objects(n_users, path="../../isaacdata/")
passrates, stretches, passquals, all_qids = load_new_diffs("../../isaacdata/pass_diffs.csv")
mcmcdf = pandas.read_csv("../../isaacdata/mcmc/dir_mcmc_results.csv", header=0, index_col=0)
qtypes = pandas.read_csv("../../isaacdata/atypes.csv", header=None, index_col=0)
predictor = pickle.load(open("p_LSVC_0.2_0.5.pkl", "rb"))
scaler = pickle.load(open("qutor_scaler.pkl", "rb"))
print("loaded data")

all_qids = list(all_qids)
random.shuffle(all_qids)

actions = tuple(all_qids)[0:25]
# qutor = Qutor(alpha=0.1, gamma=1.0, eps=1000, actions=actions)
qutor = QutorFA(1, 0.5, 1, actions, "QutorFA", cat_lookup, cat_ixs, passrates, stretches, levels, passquals)
#prep the action bank
for A in actions:
    median_runs_to_succ = mcmcdf.loc[A, "mdRTS"]
    median_runs_to_succ = numpy.mean(mcmcdf.loc[:, "mdRTS"]) if numpy.isnan(median_runs_to_succ).any() else median_runs_to_succ
    qtype = 1.0 if str(qtypes.loc[A, 7]) == "choice" else 0.0
    q = gen_qenc(median_runs_to_succ, passrates[A], stretches[A], levels[A], qtype)
    qutor.qencs[A] = gen_qenc(median_runs_to_succ, passrates[A], stretches[A], levels[A], qtype)
print("qenc cache populated")

sprofs = pandas.read_csv("../../isaacdata/student_profiling/users_all.csv", header=0, index_col=0)
sprofs = sprofs[sprofs["role"] == "STUDENT"]
sprofs = sprofs[sprofs["date_of_birth"].notna()]
sprofs = sprofs[sprofs.index.isin(users)]
users = sprofs.index

# qutor.s = K
print("init'd Qutor")

print("starting loops...")

n_trials = 50000
n_lessons = 5
#scores = pandas.DataFrame(index=range(n_trials), columns=["score"])
scores = []
#for x in range(n_trials):
score = 0
x = 0
vic = 0
while vic < 100:
    print("\nstudent {}, eps{}".format(x, qutor.EPS))
    student = StudentSim(predictor, scaler)
    K = numpy.zeros(shape=(itemencoding.n_components, itemencoding.k_features))  # K33 vector encoding
    S = numpy.zeros(shape=itemencoding.s_features)
    qtype = numpy.zeros(shape=1)

    Rtot = 0
    explorcnt = 0
    reps = 0

    alpha= 0.2
    phi= 0.5
    passed = False

    gender = 0
    age = 17
    u_def_lev = 3

    S[0] = gender
    S[1] = age
    S[2] = u_def_lev

    X = student.encode_student(S,K)
    score = 0
    for i in range(n_lessons):  # what score can we get in 100 moves?
        # print("X sum ...", numpy.sum(X))

        A, explorative = qutor.choose_A(X)
        if explorative:
            explorcnt += 1
        # print(A, explorative)

        qenc = qutor.qencs[A]

        if A in student.havedone:
            reps += 1

        R=0
        if student.doipass(A, X, qenc) == True:
            # print("*** S U C C E S S ***")
            print("S", end="")
            passed = True
            R = 1
        else:
            print("f", end="")
            # print("- - -f-a-i-l")
            passed = False
            R = -1

        score+=R
        if score == 5:
            vic+=1
        x+=1
        #print(K)
        xX = numpy.copy(X)
        K,S = gen_X_primed(K, S, cat_ixs[cat_lookup[A]], alpha, phi, passed, passrates[A], stretches[A], levels[A])
        X = student.encode_student(S,K)
        qutor.sa_update(xX, A, R, X, i==(n_lessons-1))
    qutor.replay(32)
    # scores.loc[x,"score"] = score
    scores.append(score)

print("plotting")
sx = pandas.DataFrame(index=range(len(scores)), columns=["score"])
sx.loc[:,"score"] = scores
sx.plot()
plt.show()