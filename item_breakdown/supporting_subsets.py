import random

import pandas
from matplotlib import pyplot as plt
import numpy
import pandas as pd
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from backfit.BackfitUtils import init_objects
from utils.utils import extract_runs_w_timestamp

n_users = 200
base = "../../../isaac_data_files/"
qtypes = pd.read_csv(base+"atypes.csv", header=None, index_col=0)
cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users, seed=666)
all_qids = list(qtypes.index)

ssubsets = numpy.zeros(shape=(qtypes.shape[0], qtypes.shape[0])) # these will become the supporting subsets
setcnts = numpy.zeros(shape=qtypes.shape[0])

gen = True

if gen:
    for u in users:
        X = numpy.zeros(shape=qtypes.shape[0])
        print("user = ", u)
        attempts = pd.read_csv(base+"by_user/{}.txt".format(u), header=None)
        runs = extract_runs_w_timestamp(attempts)
        for run in runs:
            ts, q, n_atts, n_pass = run
            qt = q.replace("|", "~")
            lev = levels[qt]
            qix = all_qids.index(qt)
            if n_pass:
                ssubsets[qix,:] = ssubsets[qix,:]+X
                X[qix]+=1
            setcnts[qix]+=1
    ssubsets = ssubsets / setcnts.reshape((1,-1))
    print("saving subsets")
    sdf = pandas.DataFrame(ssubsets)
    sdf.index=all_qids
    #numpy.savetxt(base+"ssubsets.csv", ssubsets, delimiter=",", fmt="%1i")
    sdf.to_csv(base+"ssubsets.csv")

else:
    print("starting plotting")
    #ssets = numpy.loadtxt(base+"ssubsets.csv", delimiter=",")
    ssets = pandas.read_csv(base+"ssubsets.csv", header=0, index_col=0)#.iloc[:,0:100]

    print("file loaded with {} rows".format(ssets.shape[0]))

    ssets.fillna(0,inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(ssets) #.as_matrix())
    print("scaled")

    affinity_propagation = cluster.AffinityPropagation() #damping=0.9, preference=-500)
    y_pred = affinity_propagation.fit_predict(ssets)
    print("clustered")

    X_embedded = TSNE(n_components=2).fit_transform(ssets)
    # X_embedded = PCA(n_components=2).fit_transform(ssets)
    print(X_embedded.shape)
    print("dim reduced")

    centroids = affinity_propagation.cluster_centers_indices_
    print(centroids)
    centroid_arr = numpy.zeros(shape=len(y_pred))
    cxlabels = []
    for cx in centroids:
        centroid_arr[cx]=1
        x = X_embedded[cx, 0]
        y = X_embedded[cx, 1]
        cxlabels.append((cx,(x,y)))

    r = lambda: random.randint(0,255)
    colours = numpy.array(['#%02X%02X%02X' % (r(),r(),r()) for i in range(max(y_pred) + 1)])
    print(y_pred)
    print(colours)

    ssets.insert(0,"cluster",y_pred)
    ssets(0,"centroid",centroid_arr)
    ssets.to_csv(base + "ssubsets_with_clusters.csv")

    r = lambda: random.randint(0,255)
    colours = numpy.array(['#%02X%02X%02X' % (r(),r(),r()) for i in range(max(y_pred) + 1)])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.5, c=colours[y_pred])
    for lab in cxlabels:
        txt = ssets.index[ lab[0] ].split("~")[0]
        coords = lab[1]
        plt.annotate(txt, coords)

    plt.title("tSNE map of questions, clustered by supporting set")
    plt.show()
    exit()

    set_sums = ssets.sum(axis=1)
    #print(pandas.DataFrame(set_sums).describe())
    while(True):
        ix = random.randint(0,ssets.shape[0]-1)
        qid=ssets.index[ix]
        plt.bar( range(ssets.shape[0]), list(ssets.iloc[ix,:]) )
        plt.title("Supporting set for {}".format(qid))
        plt.show()