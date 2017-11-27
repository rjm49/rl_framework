import os, sys

from sklearn.naive_bayes import BernoulliNB

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backfit.BackfitUtils import init_objects
from backfit.utils.utils import DW_STRETCH, DW_LEVEL, calc_qdiff, load_new_diffs, DW_NO_WEIGHT, DW_BINARY, DW_NATTS, \
    DW_PASSRATE, load_mcmc_diffs, DW_MCMC
from backfit.BackfitTest import train_and_test

print(sys.path)
from sklearn.metrics.classification import f1_score
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import SVC, LinearSVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.dummy import DummyClassifier

import pandas as pd
import numpy
from utils.utils import extract_runs_w_timestamp

QENC_QUAL=False
QENC_DIFF=False
qenc_width = 33
n_classes = 2

FEAT_F33 = "F33"

n_users = 1000
max_runs = None #10000
percTest = 0.10

predictors = [
    # DummyClassifier(strategy="stratified"),
    # DummyClassifier(strategy="uniform"),
    # BernoulliNB(),
    #SVC(max_iter=1000, class_weight="balanced"),
    LinearSVC(max_iter=1000, class_weight="balanced"),
    MLPClassifier(max_iter=100, nesterovs_momentum=True, early_stopping=True), #, activation="logistic"),
    # LogisticRegression(class_weight='balanced'),
    # GaussianNB(),
]

predictor_params = [
    # None,
    # None,
    #{'n_iter':50, 'alpha': numpy.logspace(-3, 2) },
    #{'n_iter':50,'C': numpy.logspace(-3, 2), 'gamma': numpy.logspace(-3, 2)},
    {'n_iter':50,'C': numpy.logspace(-3, 2)},
    {'n_iter':50,'hidden_layer_sizes':[(442,),], 'learning_rate_init':[0.001, 0.01, 0.1], 'alpha': numpy.logspace(-6,2) },
    # {'n_iter':50,'C': numpy.logspace(-3, 2)},
    # None,
]


def generate_run_files(alpha, _featureset_to_use, _w, fade, cats, cat_lookup, all_qids, users, stretches, passrates, passquals, levels, mcmcdiffs, cat_ixs):
    stem = _featureset_to_use+"_"+str(alpha) + "_" + str(fade) + "_" + _w
    x_filename= stem+"_X.csv"
    y_filename= stem+"_y.csv"

    X_file = open(stem+"_X.csv","w")
    y_file = open(stem+"_y.csv","w")

    n_components = len(cats)
    n_features = 14
    #     all_X = numpy.zeros(shape=(0,n_features))

    print("using n_features=", n_features)

    # tmx = numpy.loadtxt("../mcmc/X.csv", delimiter=",") # load the prob transition mx
    # qf = open("../mcmc/obsqs.txt")
    # qindex = [rec.split(",")[0] for rec in qf.read().splitlines()]
    # qf.close()
    #
    # print(tmx.shape[0], len(qindex))
    # assert tmx.shape[0] == len(qindex)
    # print("loaded transition data")

    run_ct= 0
    X = numpy.zeros(shape=(n_components, n_features) ) #init'se a new feature vector w same width as all_X
    print("Generating files for {} users...".format(len(users)))
    for u in users:
        print("user = ", u)
        X[:]= 0.0

        attempts = pd.read_csv("../../isaacdata/by_user/{}.txt".format(u), header=None)

        runs = extract_runs_w_timestamp(attempts)
        for run in runs:
            run_ct+=1
            ts, q, n_atts, n_pass = run
            qt = q.replace("|","~")
            lev = levels[qt]
            if lev<1:
                continue

            # qdiff = calc_qdiff(qt, passrates, stretches, levels, mcmcdiffs, mode=_w)

            catix = cat_ixs[ cat_lookup[qt] ]

            passrate = passrates[qt]
            qpassqual = passquals[qt]
            stretch = stretches[qt]
            mcmc = mcmcdiffs[qt] if qt in mcmcdiffs else 0

            qenc = numpy.zeros(shape=(qenc_width, n_features))

            PRATE_IX=0
            STRETCH_IX=1
            NATTS_IX=2
            LEVEL_IX=3
            MCMC_IX=4
            QUAL_IX=5
            SUXX_IX=6
            Q_CNT_IX=7
            RECENCY_IX=8
            FAIL_IX=9

            F_PRATE_IX=10
            F_STRETCH_IX=11
            F_NATTS_IX=12
            F_QUAL_IX=13


            qenc[catix, PRATE_IX] = passrate
            qenc[catix, STRETCH_IX] = stretch
            qenc[catix, NATTS_IX] = n_atts
            qenc[catix, LEVEL_IX] = lev
            # qenc[catix, MCMC_IX] = mcmc
            qenc[catix, QUAL_IX] = qpassqual

            X_file.write(",".join([str(x) for x in X.flatten()])+","+",".join([str(e) for e in qenc.flatten()])+"\n")

            X[catix, RECENCY_IX] = X[catix, RECENCY_IX] * fade
            X[catix, Q_CNT_IX] += 1.0


#            X[catix, LEVEL_IX] += alpha*lev
#            X[catix, MCMC_IX] += alpha*mcmc

            if (n_pass>0):
                y = -1 if (n_classes==3 and n_atts==1) else 0
                X[catix, PRATE_IX] = X[catix, PRATE_IX]*(1-alpha) + alpha*passrate
                X[catix, QUAL_IX] = X[catix, QUAL_IX]*(1-alpha) + alpha*qpassqual
                X[catix, STRETCH_IX] = X[catix, STRETCH_IX]*(1-alpha) + alpha*stretch
                X[catix, NATTS_IX] = X[catix, NATTS_IX]*(1-alpha) + alpha*n_atts

                X[catix, SUXX_IX] += 1.0
                X[catix, RECENCY_IX] = 1.0
            else:
                y = 1
                X[catix, F_PRATE_IX] = X[catix, F_PRATE_IX]*(1-alpha) + alpha*passrate
                X[catix, F_QUAL_IX] = X[catix, F_QUAL_IX]*(1-alpha) + alpha*qpassqual
                X[catix, F_STRETCH_IX] = X[catix, F_STRETCH_IX]*(1-alpha) + alpha*stretch
                X[catix, F_NATTS_IX] = X[catix, F_NATTS_IX]*(1-alpha) + alpha*n_atts

                X[catix, FAIL_IX] += 1.0
                X[catix, RECENCY_IX] = -1.0

            y_file.write(str(y)+"\n")

        X_file.flush()
        y_file.flush()
    X_file.close()
    y_file.close()
    print(n_users, "users", run_ct,"runs", run_ct/float(n_users), "rpu")
    return x_filename,y_filename


if __name__ == '__main__':
    featureset_to_use=FEAT_F33
    cmd='test'
    if len(sys.argv) < 2:
        cmd = input("command please?")
    else:
        cmd = sys.argv[1]

    if cmd.startswith('g'):
        do_test = False
    else:
        do_test = True

    force_balanced_classes = True
    do_scaling = True
    optimise_predictors = True
    n_classes = 2
    print("n_users",n_users)
    cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users, seed=666)

    #users = open("../mcmc/mcmc_uesrs.txt").read().splitlines()

    passdiffs, stretches, passquals, all_qids = load_new_diffs("../../isaacdata/pass_diffs.csv")
    mcmcdiffs = load_mcmc_diffs("../../isaacdata/mcmc/mcmc_results.csv")

    reports =[]
    report_name = "qutorgen_{}_{}_fb{}_opt{}_scale{}_{}.txt".format(0, n_users, str(1 if force_balanced_classes else 0), ("001" if optimise_predictors else "0"), ("1" if do_scaling else "0"), featureset_to_use)
    if do_test:
        report = open(report_name,"w")
    for w in [DW_BINARY]: #DW_NO_WEIGHT, DW_NATTS, DW_LEVEL, DW_PASSRATE, DW_MCMC, DW_STRETCH]:
        for alpha in [0.25, 0.5, 0.75, 1.0]:
            for phi_retain in [1.0, 0.75, 0.5, 0.25, 0.0]:
                print(cat_ixs)
                if do_test:
                    print("testing")
                    xfn = "F33_{}_{}_{}_X.csv".format(str(alpha), str(phi_retain), w)
                    yfn = "F33_{}_{}_{}_y.csv".format(str(alpha), str(phi_retain), w)
                    X_train, X_test, y_pred_tr, y_pred, y_true, scaler = train_and_test(alpha, predictors, predictor_params, xfn, yfn, n_users, percTest, featureset_to_use, w, phi_retain, force_balanced_classes, do_scaling, optimise_predictors, report=report)
                    #reports.append((alpha, report_name, y_true, y_pred))
                else:
                    xfn, yfn = generate_run_files(alpha, featureset_to_use, w, phi_retain, cats, cat_lookup, all_qids, users, stretches, passdiffs, passquals, levels, mcmcdiffs, cat_ixs)
                    print("gen complete, train files are",xfn,yfn)
    if do_test:
        report.close()
        print("complete, report file is:", report_name)


        # wid = n_classes+1
        # if do_test:
        #     retains = []
        #     f1s = []
        #     mx = numpy.ndarray(shape=(len(reports), wid))
        #     for ix, (alpha, r, ytr, ypd) in enumerate(reports):
        #         mx[ix, 0] = alpha
        #         f1s = f1_score(ytr, ypd, average=None)
        #         mx[ix, 1:wid] = f1s
        #     numpy.savetxt(report_name+"_to_plot.csv", mx)