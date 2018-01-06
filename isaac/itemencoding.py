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

n_components = 33
s_features = 9
q_features = 4
k_features = 3

def gen_qenc(catix, median_xp_to_s, passrate, stretch, lev, qpassqual, catsxp):
    # qx = numpy.zeros(shape=(1,))
    #qx[0] = numpy.sum(catsxp)
    # qx[0] = stretch
    qenc = numpy.zeros(shape=q_features)
    qenc[0] = (lev+1)
    qenc[1] = passrate
    qenc[2] = stretch
    qenc[3] = median_xp_to_s
    # qenc[catix, MCMC_IX] = mcmc
    #qenc[catix, 3]    = qpassqual
    #print(qenc.shape)
    #print(catsxp.shape)
    #qenc = numpy.hstack((qenc, catsxp.reshape(-1,1)))
    #print(qenc.shape)
    #qenc[:,3] = catsxp.reshape(33,)
    #qenc = numpy.append(qx, qenc.flatten())
    return qenc

SS_SLEV_IX = 3
def gen_X_primed(X, S, catix, alpha, fade, is_pass, passrate, qpassqual, stretch, lev, catsxp, n_atts):
    #semi static traits
    SS_OPLEV_IX = 3
    SS_FLEV_IX = 4
    SS_XP_IX = 5
    SS_SXP_IX = 6
    SS_FXP_IX = 7
    SS_SMAX_IX = 8

    LEVEL_IX=0
    # F_LEVEL_IX=2
    PRATE_IX=1
    STRETCH_IX=2
    F_PRATE_IX=3
    F_STRETCH_IX=4
    # RECENCY_IX =0
    # F_RECENCY_IX=0
    # NATTS_IX=0
    # F_NATTS_IX=0
    # SUXX_IX=0
    # FAIL_IX=0
    # XP = 0
    # SXPAV = 0
    # F_SXPAV = 0

    Xcat = X[catix]
    Xcat = Xcat * fade
    # X[:,RECENCY_IX]  =   X[:,RECENCY_IX] * fade
    # X[:, F_RECENCY_IX] = X[:, F_RECENCY_IX] * fade
    # X[catix, Q_CNT_IX]    +=  1.0
    #            X[catix, LEVEL_IX] += alpha*lev
    #            X[catix, MCMC_IX] += alpha*mcmc
    # Xcat[XP] = Xcat[XP] + 1.0
    S[SS_XP_IX] = S[SS_XP_IX] + lev+1
    #S[SS_OPLEV_IX] = S[SS_OPLEV_IX] * (1 - alpha) + alpha * (lev+1)
    if (is_pass):
        # print("passed",passrate, stretch, lev)
        ix = ["PRATE_IX", "STRETCH_IX", "LEVEL_IX"]

        # X[:, SXPAV] = X[:, SXPAV]*(1 - alpha) + alpha * catsxp.reshape(33,)
        # print(X[:,SXPAV].shape)
        S[SS_SLEV_IX] = S[SS_SLEV_IX] * (1 - alpha) + alpha * (lev+1)
        S[SS_SXP_IX] = S[SS_SXP_IX] + lev+1
        S[SS_SMAX_IX] = max(S[SS_SMAX_IX], (lev+1))
        Xcat[LEVEL_IX] = Xcat[LEVEL_IX] * (1 - alpha) + alpha * (lev+1)
        Xcat[STRETCH_IX]  = Xcat[STRETCH_IX] * (1 - alpha) + alpha * stretch
        Xcat[PRATE_IX]    = Xcat[PRATE_IX] * (1 - alpha) + alpha * passrate

        #Xc[QUAL_IX]     = Xc[QUAL_IX]   * (1 - alpha) + alpha * qpassqual
        # Xcat[NATTS_IX]    = Xcat[NATTS_IX]  * (1 - alpha) + alpha * n_atts
        # Xcat[SUXX_IX]     += 1.0
        #Xcat[RECENCY_IX]  = 1.0
    else:
        S[SS_FXP_IX] = S[SS_FXP_IX] + lev+1
        S[SS_FLEV_IX] = S[SS_FLEV_IX] * (1 - alpha) + alpha * (lev+1)
        # X[:, F_SXPAV] = X[:, F_SXPAV]*(1 - alpha) + alpha * catsxp.reshape(33,)
        #Xcat[F_PRATE_IX]  = Xcat[F_PRATE_IX] * (1 - alpha) + alpha * passrate
        #Xcat[F_STRETCH_IX] = Xcat[F_STRETCH_IX] * (1 - alpha) + alpha * stretch
        # Xcat[F_LEVEL_IX]  = Xcat[F_LEVEL_IX] * (1 - alpha) + alpha * lev
        #Xc[F_QUAL_IX]   = Xc[F_QUAL_IX]     * (1 - alpha) + alpha * qpassqual
        # Xcat[F_NATTS_IX]  = Xcat[F_NATTS_IX]    * (1 - alpha) + alpha * n_atts
        # Xcat[FAIL_IX]     += 1.0
        #Xcat[F_RECENCY_IX]  = 1.0
    X[catix] = Xcat
    # print("X'=",X)
    return X, S