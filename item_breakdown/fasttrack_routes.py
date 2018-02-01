import random

import pandas
from matplotlib import pyplot as plt
import numpy
import pandas as pd
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler

from backfit.BackfitUtils import init_objects
from utils.utils import extract_runs_w_timestamp
from mpl_toolkits.mplot3d import Axes3D
from leven import levenshtein

n_users = -1
base = "../../../isaac_data_files/"
qtypes = pd.read_csv(base+"atypes.csv", header=None, index_col=0)
cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users, seed=666)

uxm = None
all_qids = open("./old_fasttrack.txt","r").read().splitlines()

gen = True

def get_rand():
    return random.randint(255)

ft_users=[]
if gen:
    for u in users:
        print("user = ", u)
        attempts = pd.read_csv(base+"by_user/{}.txt".format(u), header=None)
        runs = extract_runs_w_timestamp(attempts)
        for run in runs:
            ts, q, n_atts, n_pass = run
            qt = q.replace("|", "~")
            if(qt in all_qids and u not in ft_users):
                ft_users.append(u)

    att_arr=[]
    scr_arr=[]
    for u in ft_users:
        cnt=1
        att=0
        scr=0
        X = numpy.zeros(shape=len(all_qids))
        print("user = ", u)
        attempts = pd.read_csv(base+"by_user/{}.txt".format(u), header=None)
        runs = extract_runs_w_timestamp(attempts)
        for run in runs:
            ts, q, n_atts, n_pass = run
            qt = q.replace("|", "~")
            lev = levels[qt]
            att+=1
            if(qt in all_qids):
                qix = all_qids.index(qt)
                if n_pass:
                    X[qix]=cnt
                    cnt+=1
                    scr+=1
        att_arr.append(att)
        scr_arr.append(scr)
        if uxm is None:
            uxm=X
        else:
            uxm = numpy.vstack((uxm,X))
    print("saving fasttrack routes")
    udf = pandas.DataFrame(uxm)
    udf.index=ft_users
    udf.insert(0,"attempts",att_arr)
    udf.insert(0,"score",scr_arr)
    udf.to_csv(base+"ft_routes.csv")
    #numpy.savetxt(base+"ssubsets.csv", ssubsets, delimiter=",", fmt="%1i")

else:
    print("start plotting")
    #ssets = numpy.loadtxt(base+"ssubsets.csv", delimiter=",")
    ssets = pandas.read_csv(base+"ft_routes.csv", header=0, index_col=0)
    print("data loaded")
    ssets.fillna(0,inplace=True)
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(ssets.as_matrix())
    X_scaled = ssets

    affinity_propagation = cluster.AffinityPropagation() #damping=0.9)
    y_pred = affinity_propagation.fit_predict(X_scaled)

    X_embedded = TSNE(n_components=2).fit_transform(X_scaled)
    # X_embedded = PCA(n_components=2).fit_transform(X_scaled)
    print(X_embedded.shape)
    print("Dim reduce complete")

    centroids = affinity_propagation.cluster_centers_indices_
    print(centroids)
    centroid_arr = numpy.zeros(shape=len(y_pred))
    cxlabels = []
    for cx in centroids:
        centroid_arr[cx]=1
        x = X_embedded[cx, 0]
        y = X_embedded[cx, 1]
        cxlabels.append((cx,(x,y)))

    print("clustering complete")

    ssets.insert(0,"cluster",y_pred)
    ssets.insert(0,"centroid",centroid_arr)
    ssets.sort_values("cluster", inplace=True)
    ssets.to_csv(base + "ft_routes_clusters.csv")
    print("cluster list saved")

    r = lambda: random.randint(0,255)
    colours = numpy.array(['#%02X%02X%02X' % (r(),r(),r()) for i in range(max(y_pred) + 1)])
    print(y_pred)
    print(colours)

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    # ax = fig.gca(projection='3d')
    # ax = fig.add_subplot(111, projection='3d')

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.5, c=colours[y_pred])
    # ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], alpha=0.5, c=colours[y_pred])
    for lab in cxlabels:
        txt = ssets.index[ lab[0] ]
        coords = lab[1]
        plt.annotate(txt, coords)

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