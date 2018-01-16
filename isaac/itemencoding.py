import numpy

# PRATE_IX = 0
# STRETCH_IX = 1
# NATTS_IX = 2
# LEVEL_IX = 3
# MCMC_IX = 4
# QUAL_IX = 5
# SUXX_IX = 6
# Q_CNT_IX = 7
# RECENCY_IX = 8
#
# FAIL_IX = 9
# F_PRATE_IX = 10
# F_STRETCH_IX = 11
# F_NATTS_IX = 12
# F_QUAL_IX = 13
import pandas


n_components = 6060
s_features = 1
q_features = 33
k_features = 1

student_state_width = n_components+s_features
X_width = n_components+s_features+q_features+1

def create_S(S, sprofs, ts, u, u_start_age):
    gen = sprofs.loc[int(u), "gender"]
    ts = pandas.to_datetime(ts)
    u_def_lev = sprofs.loc[int(u), "default_level"]
    assert isinstance(u_def_lev, numpy.float64)
    u_def_lev = 0 if numpy.isnan(u_def_lev) else (u_def_lev + 1)
    # print(ts)
    dob_ts = pandas.to_datetime(sprofs.loc[int(u), "date_of_birth"])
    # print(dob_ts)
    age_delta = (ts - dob_ts)
    min_age = pandas.Timedelta(14 * 365.242, unit='d')
    max_age = pandas.Timedelta(20 * 365.242, unit='d')
    age_delta = max_age if age_delta > max_age else max(age_delta, min_age)
    age = age_delta.days / 365.242
    if u_start_age is None:
        u_start_age = age_delta
    print(age)
    # print(age, age.seconds, age.days/365.242, type(age))
    # exit()
    # S[0] = 0 if gen == "MALE" else 1
    # S[1] = age
    # S[2] = u_def_lev
    return S, age, u_def_lev, u_start_age

def gen_qenc(catix, median_xp_to_s, passrate, stretch, lev, qtype):
    qenc = numpy.zeros(shape=q_features)
    qenc[catix] = (lev+1)
    qenc = numpy.append(qenc, qtype)
    return qenc

SS_SLEV_IX = 3
def gen_X_primed(X, S, qix, alpha, fade, is_pass, passrate, stretch, lev):
    X[qix] = 1 if is_pass else -1
    return X, S
