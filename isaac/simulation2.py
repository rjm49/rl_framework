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
from isaac.dqtutor import DQTutor
from isaac.itemencoding import create_S, X_width, student_state_width

target = 1000
n_users = 1000
# random.seed(666)
scores = []

cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs = init_objects(n_users, path="../../../isaac_data_files/")
passrates, stretches, passquals, all_qids = load_new_diffs("../../../isaac_data_files/pass_diffs.csv")
mcmcdf = pandas.read_csv("../../../isaac_data_files/mcmc/dir_mcmc_results.csv", header=0, index_col=0)
qtypes = pandas.read_csv("../../../isaac_data_files/atypes.csv", header=None, index_col=0)
predictor = pickle.load(open("../../../isaac_data_files/p_RFOR_1.0_1.0.pkl", "rb"))
rdim = pickle.load(open("../../../isaac_data_files/qutor_rdim.pkl", "rb"))
scaler = pickle.load(open("../../../isaac_data_files/qutor_scaler.pkl", "rb"))
print("loaded data")

all_qids = list(all_qids)
random.shuffle(all_qids)

n_actions = 12

actions = tuple(all_qids)[0:n_actions]
# qutor = Qutor(alpha=0.1, gamma=1.0, eps=1000, actions=actions)
dqutor = DQTutor(student_state_width,n_actions)
qencs = {}
#prep the action bank
for aix,A in enumerate(actions):
    median_runs_to_succ = mcmcdf.loc[A, "mdRTS"]
    median_runs_to_succ = numpy.mean(mcmcdf.loc[:, "mdRTS"]) if numpy.isnan(median_runs_to_succ).any() else median_runs_to_succ
    catix = cat_ixs[cat_lookup[A]]
    qtype = 1.0 if str(qtypes.loc[A, 7]) == "choice" else 0.0
    qenc = gen_qenc(catix, median_runs_to_succ, passrates[A], stretches[A], levels[A], qtype)
    qencs[aix] = (A, qenc)
print("qenc cache populated")

sprofs = pandas.read_csv("../../../isaac_data_files/student_profiling/users_all.csv", header=0, index_col=0)
sprofs = sprofs[sprofs["role"] == "STUDENT"]
sprofs = sprofs[sprofs["date_of_birth"].notna()]
sprofs = sprofs[sprofs.index.isin(users)]
users = sprofs.index

# qutor.s = K
print("init'd Qutor")

print("starting loops...")

n_trials = 10000
dqutor.explore_period = 20000
n_lessons = 4
scores = pandas.DataFrame(index=range(n_trials), columns=["score","return","reps"])
end = False
for x in range(n_trials):
    print("\nstudent {}, eps{}".format(x, dqutor.epsilon))
    student = StudentSim(predictor, rdim, scaler)
    K = numpy.zeros(shape=(itemencoding.n_components, itemencoding.k_features))  # K33 vector encoding
    S = numpy.zeros(shape=itemencoding.s_features)
    qtype = numpy.zeros(shape=1)

    Rtot = 0
    explorcnt = 0
    reps = 0

    alpha= 1.0
    phi= 1.0
    passed = False

    gender = 0
    age = 17
    u_def_lev = 3

    # S[0] = gender
    # S[1] = age
    # S[2] = u_def_lev

    lssns = []
    X = student.encode_student(S,K)
    score = 0
    Rtot = 0
    for i in range(n_lessons):  # what score can we get in 100 moves?
        print("X sum ...", numpy.sum(X))

        Aix, exp = dqutor.act(X)
        # print("action is",Aix)
        A, qenc = qencs[Aix]
        # print("****")
        # print(X)
        # print(qenc)
        # print("****")
        # print(X.shape, qenc.shape)

        R=0

        if A in student.havedone:
            print("   dupe ")
            reps += 1
            R = -50
        elif student.doipass(A, X, qenc) == True:
            print("S{}".format("!" if exp==True else "."), end="")
            passed = True
            R = 5
            score += 10
        else:
            print("f{}".format("!" if exp==True else "."), end="")
            passed = False
            R = -1

        #print(K)
        # print("copying Xx")
        Rtot += R
        xX = numpy.copy(X)
        # print("gen'g X'")
        K,S = gen_X_primed(K, S, cat_ixs[cat_lookup[A]], alpha, phi, passed, passrates[A], stretches[A], levels[A])
        # print("encdoing X")
        X = student.encode_student(S,K)
        # qutor.sa_update(xX, A, R, X)
        # if i==n_lessons-1:
        #     R=score
        #     end = True
        # print("updating Q")
        dqutor.updateQ(xX, Aix, R, X, end)
        dqutor.remember(xX, Aix, R, X, end)
        lssns.append(Aix)
        # print("replay 32")
        #dqutor.replay(32)
    print(" ", score, lssns, reps)
    scores.loc[x,["score","return","reps"]] = [score,Rtot,reps]

print("plotting")
scores.plot()
plt.show()