import numpy
import pandas as pd

from backfit.BackfitUtils import init_objects
from utils.utils import extract_runs_w_timestamp

n_users = 1000
base = "../../../isaac_data_files/"
qtypes = pd.read_csv(base+"atypes.csv", header=None, index_col=0)
cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users, seed=666)
all_qids = list(qtypes.index)

ssubsets = numpy.ones(shape=(qtypes.shape[0], qtypes.shape[0])) # these will become the supporting subsets

#FOR EACH STUDENT, GET:
#start/end age
#days on platform
#qpd
#len of journey (runs)
#bucketted q counts by level
#median level
#subject mix

students = pd.read_csv(base+"infer/users_all.csv", header=0, index_col=0, sep=",")
# students = udf[(udf["role"] == "STUDENT") & (udf["date_of_birth"].notnull())]
dobs = students['date_of_birth']

udf = pd.DataFrame(index=users, columns=["REG","S_AGE","E_AGE","DAYS","LEN","QPD","MEANLEV","N_SUBJ"])

for s in dobs.index:
    print(s, type(s))

for u in users:
    print(u)
    if int(u) in (dobs.index):
        dob = pd.to_datetime( dobs[int(u)] )
    q_cnt = 0
    ts_start = 0
    ts_end = 0
    ulevs = []
    usubjs = set()
    print("user = ", u)
    attempts = pd.read_csv(base+"by_user/{}.txt".format(u), header=None)
    runs = extract_runs_w_timestamp(attempts)
    for run in runs:
        q_cnt+=1
        ts, q, n_atts, n_pass = run
        if not ts_start:
            ts_start = ts
            if pd.notna(dob):
                age_start = (pd.to_datetime(ts_start)-dob).total_seconds() / (86400 * 365.2425)
                udf.loc[u,"REG"]=ts_start
                udf.loc[u,"S_AGE"]=age_start
        qt = q.replace("|", "~")
        qix = all_qids.index(qt)
        qrow = qtypes.iloc[qix, :]
        subj = "/".join(map(str, qrow[[2, 3, 4]]))
        lev = levels[qt]
        ulevs.append(lev)
        usubjs.add(subj)
    ts_end = ts
    if pd.notna(dob):
        age_end = (pd.to_datetime(ts_end) - dob).total_seconds() / (86400 * 365.2425)
        udf.loc[u, "E_AGE"] = age_end
    days = (pd.to_datetime(ts_end) - pd.to_datetime(ts_start)).days +1
    udf.loc[u,"DAYS"] = days
    udf.loc[u,"LEN"] = q_cnt
    if days:
        udf.loc[u,"QPD"] = q_cnt/days
    udf.loc[u,"N_SUBJ"]=len(usubjs)
    udf.loc[u,"MEANLEV"]=numpy.mean(ulevs)

print("saving data")
udf.to_csv(base+"user_profiles.csv")

