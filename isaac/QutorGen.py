import heapq
import os, sys
import random

import matplotlib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier

from isaac.itemencoding import gen_qenc, gen_X_primed, k_features, s_features, SS_SLEV_IX, create_S

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backfit.BackfitUtils import init_objects
from backfit.utils.utils import DW_STRETCH, DW_LEVEL, calc_qdiff, load_new_diffs, DW_NO_WEIGHT, DW_BINARY, DW_NATTS, \
    DW_PASSRATE, load_mcmc_diffs, DW_MCMC, load_atypes
from backfit.BackfitTest import train_and_test

print(sys.path)
from sklearn.metrics.classification import f1_score
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import SVC, LinearSVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.dummy import DummyClassifier
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')

import pandas as pd
import numpy
from utils.utils import extract_runs_w_timestamp

QENC_QUAL=False
QENC_DIFF=False

n_classes = 2

FEAT_F33 = "F33"
ROLL_LEN = 10
PANTHEON_SIZE= 10

n_users = 1000
max_runs = None #10000
percTest = 0.1

predictors = [
    # DummyClassifier(strategy="stratified"),
    # DummyClassifier(strategy="uniform"),
    # BernoulliNB(),
    # SVC(kernel='rbf', max_iter=10000, class_weight="balanced", verbose=1),
    LinearSVC(max_iter=50),
    MLPClassifier(max_iter=50, nesterovs_momentum=True, early_stopping=True), #, activation="logistic"),
    # LogisticRegression(class_weight='balanced'),
    RandomForestClassifier(class_weight="balanced"),
     # ExtraTreeClassifier(),
    # AdaBoostClassifier(),
    # DecisionTreeClassifier(),
]

predictor_params = [
    # None,
    # None,
    #{'n_iter':50, 'alpha': numpy.logspace(-3, 2) },
    # {'name':'RBFSVC', 'n_iter':50,'C': numpy.logspace(-2, 6), 'gamma': numpy.logspace(-9, 3)},
    {'name':'LSVC', 'n_iter':50,'C': numpy.logspace(-3, 4)},
    {'name':'MLP', 'n_iter':50,'hidden_layer_sizes':[(80,),], 'learning_rate_init':[0.001, 0.01, 0.1], 'alpha': numpy.logspace(-6,2) },
    # {'name':'LOGREG', 'n_iter':50,'C': numpy.logspace(-3, 2)},
    {'name': "RFOR"},
     # {'name':"XTREE"},
    # {'name':"ADABoost"},
    # {'name':"DTREE"},
]

def generate_run_files(alpha, _featureset_to_use, _w, phi, cats, cat_lookup, all_qids, users, stretches, passrates, passquals, levels, mcmcdf, cat_ixs, profiles=None):
    stem = _featureset_to_use+"_"+str(alpha) + "_" + str(phi) + "_" + _w
    x_filename= stem+"_X.csv"
    y_filename= stem+"_y.csv"
    uq_filename= stem+"_uq.csv"

    X_file = open(stem+"_X.csv","w")
    y_file = open(stem+"_y.csv","w")
    uq_file = open(uq_filename, "w")

    n_components = len(cats)
    #     all_X = numpy.zeros(shape=(0,n_features))

    # tmx = numpy.loadtxt("../mcmc/X.csv", delimiter=",") # load the prob transition mx
    # qf = open("../mcmc/obsqs.txt")
    # qindex = [rec.split(",")[0] for rec in qf.read().splitlines()]
    # qf.close()
    #
    # print(tmx.shape[0], len(qindex))
    # assert tmx.shape[0] == len(qindex)
    # print("loaded transition data")
    # xpcnts = pd.DataFrame.from_csv("../../isaacdata/mcmc/toplot.csv")
    # sqmx = pd.DataFrame.from_csv("../../isaacdata/mcmc/sqmx.csv")
    # sqmx = pd.DataFrame.from_csv("../../isaacdata/mcmc/sqmx.csv")
    atypes = pd.DataFrame.from_csv("../../../isaac_data_files/atypes.csv", header=None)
    all_types = list(pd.unique(atypes[7]))

    user_summary_df = pd.DataFrame(columns=["runs","age","def_lev", "start_lev", "start_lev10", "end_lev", "end_lev10","end_age", "max_lev","ts_delta","qpd","rox_delta","max_delta","pant_delta","pant_avg","rox10","roxend","pantheon"], index=users)

    run_ct= 0
    print("Generating files for {} users...".format(len(users)))
    for u in users:
        u_def_lev = None
        u_start_age = None
        u_start_lev = None
        u_start_10_lev = None
        u_end_lev = None
        u_max_lev = 0
        u_roll_sx = []
        u_start_ts = None
        u_end_ts = None
        u_pantheon = []
        u_run_ct = 0
        rox10=None
        roxend=None
        print("user = ", u)
        S = numpy.zeros(shape=s_features)
        X = numpy.zeros(shape=(n_components,k_features) ) #init'se a new feature vector w same width as all_X
        oplev = 0.0
        print(X.shape)
        # attempts = pd.read_csv("../../isaacdata/by_user/{}.txt".format(u), header=None)
        # runs = extract_runs_w_timestamp(attempts)
        # fout = open("../../isaacdata/by_runs/{}.txt".format(u), "w")
        runs = open("../../../isaac_data_files/by_runs/{}.txt".format(u)).readlines()
        u_run_ct = len(runs)
        all_zero_level = True
        for run_ix,run in enumerate(runs):
            run_ct+=1
            ts, q, n_atts, n_pass = eval(run)
            if u_start_ts is None:
                u_start_ts = pd.to_datetime(ts)
            u_end_ts = pd.to_datetime(ts)
            #fout.write(",".join(map(str,(run))) + "\n")
            # fout.write(str(run)+"\n")
            # continue
            qt = q.replace("|","~")
            lev = levels[qt]
            pass
            if(lev!=0):
                all_zero_level=False
            else:
                pass
            print(all_types)
            # atix = all_types.index( str( atypes.loc[qt,7] ) )
            # print(atix)

            # A = numpy.zeros(shape=1)
            qtype = 1.0 if str(atypes.loc[qt,7])=="choice" else 0.0

            # if lev<1:
            #     continue

            catix = cat_ixs[ cat_lookup[qt] ]

            if sprofs is not None:
                S, age, u_def_lev, u_start_age = create_S(S, sprofs, ts, u, u_start_age)

            passrate = passrates[qt]
            qpassqual = passquals[qt]
            stretch = stretches[qt]
            # print("PR/ST = ",passrate,stretch)
            # sxp = xpcnts.loc[qt,"S_XP"]
            catsxps = numpy.zeros(shape=(len(cats),))
            # for c in cats:
            #     cix = cat_ixs[c]
            # catsxps[cix]=sqmx.loc[qt, c]
            print("attempting", lev)
            median_xp_to_s = mcmcdf.loc[qt,"mdRTS"]
            qenc = gen_qenc(catix, median_xp_to_s, passrate, stretch, lev, qtype)

            #X_file.write(",".join([str(x) for x in X.flatten()])+","+",".join([str(e) for e in qenc.flatten()])+"\n")
            #print(X.shape, qenc.shape)
            #print(X.flatten().shape, qenc.flatten().shape)

            #jnd = numpy.append(numpy.append(X.flatten(), ">>>") , qenc.flatten())
            # jnd = numpy.append(S.flatten(), qenc.flatten())
            jnd = numpy.append(S.flatten(), X.flatten())
            jnd = numpy.append(jnd, qenc.flatten())
            #print(jnd.shape)

            X_file.write(",".join([str(j) for j in jnd]) +"\n")

            X, S = gen_X_primed(X, S, catix, alpha, phi, (n_pass > 0), passrate, stretch, lev)

            if run_ix==0:
                u_start_lev = lev+1

            if (n_pass > 0):
                L = (lev+1)
                u_max_lev = max(L, u_max_lev)
                if len(u_pantheon)<PANTHEON_SIZE or L >= min(u_pantheon):
                    heapq.heappush(u_pantheon, L)
                    if len(u_pantheon)>PANTHEON_SIZE:
                        heapq.heappop(u_pantheon)

                u_roll_sx.append(L)
                u_end_lev = L
                if len(u_roll_sx)<=ROLL_LEN:
                    u_start_10_lev = numpy.mean(u_roll_sx)
                    rox10 = tuple(u_roll_sx)
                else:
                    u_roll_sx.pop(0) # roll
                roxend = tuple(u_roll_sx)
                u_end_10_lev = numpy.mean(u_roll_sx)

                y = -1 if (n_classes == 3 and n_atts == 1) else 0
            else:
                y = 1

            uq_file.write("{},{},{}\n".format(u,qt,y))
            y_file.write(str(y)+"\n")

        if(len(u_roll_sx)>9 and not all_zero_level):
            ts_delta = (u_end_ts - u_start_ts).days
            u_start_age = u_start_age.days / 365.242
            u_end_age = age + (u_end_ts - u_start_ts).days / 365.242
            if ts_delta > 7:
                max_delta = None if (u_start_10_lev is None or u_end_10_lev is None) else (u_max_lev - u_start_10_lev)
                rox_delta = None if (u_start_10_lev is None or u_end_10_lev is None) else (u_end_10_lev - u_start_10_lev)
                qpd = u_run_ct / ts_delta
                user_summary_df.loc[u,:] = [u_run_ct, u_start_age, int(u_def_lev), int(u_start_lev), u_start_10_lev, int(u_end_lev), u_end_10_lev, u_end_age, int(u_max_lev), ts_delta, qpd, rox_delta, max_delta, numpy.mean(u_pantheon)-u_start_10_lev, numpy.mean(u_pantheon), str(rox10), str(roxend), str(u_pantheon)]

        # fout.close()
        X_file.flush()
        y_file.flush()
    # x = numpy.random.randn(1000)
    # plt.show()
    # exit()

    user_summary_df.dropna(inplace=True)
    user_summary_df = user_summary_df.infer_objects()
    user_summary_df.to_csv("user_summary_df.csv")
    #deltadf = user_summary_df["rox_delta"]
    # print(type(deltadf))
    # print(deltadf.head())
    # user_summary_df.loc[:,"max_lev"].plot.hist(bins=7)


    X_file.close()
    y_file.close()
    uq_file.close()
    print(n_users, "users", run_ct,"runs", run_ct/float(n_users), "rpu")
    return x_filename,y_filename




if __name__ == '__main__':
    featureset_to_use=FEAT_F33
    cmd='test'
    if len(sys.argv) < 2:
        cmd = input("command please?")
    else:
        cmd = sys.argv[1]

    if cmd.startswith('p'):
        #do plots
        user_summary_df = pd.DataFrame.from_csv("user_summary_df.csv", header=0, index_col=0)
        print(user_summary_df.shape)
        print(user_summary_df.dtypes)
        print(user_summary_df.dtypes)
        user_summary_df.iloc[:, 0:15].hist(bins="auto")
        plt.show()
        exit()

    if cmd.startswith('g'):
        do_test = False
    else:
        do_test = True

    force_balanced_classes = True
    do_scaling = True
    optimise_predictors = True
    n_classes = 2
    print("n_users",n_users)
    cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users, path="../../../isaac_data_files/", seed=666)

    #users = open("../mcmc/mcmc_uesrs.txt").read().splitlines()

    passdiffs, stretches, passquals, all_qids = load_new_diffs("../../../isaac_data_files/pass_diffs.csv")
    mcmcdf = pd.DataFrame.from_csv("../../../isaac_data_files/mcmc/dir_mcmc_results.csv")

    sprofs = pd.DataFrame.from_csv("../../../isaac_data_files/student_profiling/users_all.csv")
    sprofs = sprofs[sprofs["role"]=="STUDENT"]
    sprofs = sprofs[sprofs["date_of_birth"].notna()]
    sprofs = sprofs[sprofs.index.isin(users)]
    users = sprofs.index
    print(len(users))

    # adf = load_atypes("../../isaacdata/new_atypes.csv")
    # passdiffs = adf.loc[:,"med_passrate"]
    # stretches = adf.loc[:,"med_n_pass"] / adf.loc[:,"med_n_atts"]

    reports =[]
    report_name = "qutorgen_{}_{}_fb{}_opt{}_scale{}_{}.csv".format(0, n_users, str(1 if force_balanced_classes else 0), ("001" if optimise_predictors else "0"), ("1" if do_scaling else "0"), featureset_to_use)
    conf_report = "confusion.txt"
    if do_test:
        report = open(report_name,"w")
        conf_report = open(conf_report,"w")
    for w in [DW_BINARY]: #DW_NO_WEIGHT, DW_NATTS, DW_LEVEL, DW_PASSRATE, DW_MCMC, DW_STRETCH]:
        for alpha in [1.0]: #, 0.73, 0.37, 0.1]:
            for phi_retain in [1.0]:
                print(cat_ixs)
                if do_test:
                    print("testing")
                    xfn = "F33_{}_{}_{}_X.csv".format(str(alpha), str(phi_retain), w)
                    yfn = "F33_{}_{}_{}_y.csv".format(str(alpha), str(phi_retain), w)
                    X_train, X_test, y_pred_tr, y_pred, y_true, scaler = train_and_test(alpha, predictors, predictor_params, xfn, yfn, n_users, percTest, featureset_to_use, w, phi_retain, force_balanced_classes, do_scaling, optimise_predictors, report=report, conf_report=conf_report)
                    #reports.append((alpha, report_name, y_true, y_pred))
                else:
                    xfn, yfn = generate_run_files(alpha, featureset_to_use, w, phi_retain, cats, cat_lookup, all_qids, users, stretches, passdiffs, passquals, levels, mcmcdf, cat_ixs, profiles=sprofs)
                    print("gen complete, train files are",xfn,yfn)
    if do_test:
        conf_report.close()
        report.close()
        print("complete, report file is:", report_name)


