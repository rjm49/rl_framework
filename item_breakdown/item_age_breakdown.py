import pandas as pd

from backfit.BackfitUtils import init_objects
from utils.utils import extract_runs_w_timestamp

n_users = 1000
qtypes = pd.read_csv("../../../isaac_data_files/atypes.csv", header=None, index_col=0)
cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users, seed=666)


for u in users:
    print("user = ", u)
    X[:] = 0.0

    attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)

    runs = extract_runs_w_timestamp(attempts)
    for run in runs:
        ts, q, n_atts, n_pass = run
        qt = q.replace("|", "~")
        lev = levels[qt]
        if lev < 1:
            continue

        # qdiff = calc_qdiff(qt, passrates, stretches, levels, mcmcdiffs, mode=_w)

        catix = cat_ixs[cat_lookup[qt]]

        passrate = passrates[qt]
        qpassqual = passquals[qt]
        stretch = stretches[qt]
        mcmc = mcmcdiffs[qt] if qt in mcmcdiffs else 0