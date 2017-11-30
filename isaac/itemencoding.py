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
q_features = 4
k_features = 4# 10

def gen_qenc(catix, passrate, stretch, lev, qpassqual):
    qenc = numpy.zeros(shape=(n_components, q_features))
    qenc[catix, 0]   = passrate
    qenc[catix, 1] = stretch
    qenc[catix, 2]   = lev
    # qenc[catix, MCMC_IX] = mcmc
    qenc[catix, 3]    = qpassqual
    return qenc

def gen_X_primed(X, catix, alpha, fade, is_pass, passrate, qpassqual, stretch): #, n_atts):

    RECENCY_IX=0
    Q_CNT_IX=1
    PRATE_IX=2
    QUAL_IX=0
    STRETCH_IX=0
    SUXX_IX=0
    F_PRATE_IX=3
    F_QUAL_IX=0
    F_STRETCH_IX=0
    FAIL_IX=0


    Xc = X[catix]
    X[:,RECENCY_IX]  =   X[:,RECENCY_IX] * fade
    Xc[Q_CNT_IX]    +=  1.0
    #            X[catix, LEVEL_IX] += alpha*lev
    #            X[catix, MCMC_IX] += alpha*mcmc
    if (is_pass):
        #Xc[PRATE_IX]    = Xc[PRATE_IX]  * (1 - alpha) + alpha * passrate
        #Xc[QUAL_IX]     = Xc[QUAL_IX]   * (1 - alpha) + alpha * qpassqual
        #Xc[STRETCH_IX]  = Xc[STRETCH_IX] * (1 - alpha) + alpha * stretch
        #Xc[NATTS_IX]    = Xc[NATTS_IX]  * (1 - alpha) + alpha * n_atts
        #Xc[SUXX_IX]     += 1.0
        Xc[RECENCY_IX]  = 1.0
    else:
        #Xc[F_PRATE_IX]  = Xc[F_PRATE_IX]    * (1 - alpha) + alpha * passrate
        #Xc[F_QUAL_IX]   = Xc[F_QUAL_IX]     * (1 - alpha) + alpha * qpassqual
        #Xc[F_STRETCH_IX] = Xc[F_STRETCH_IX] * (1 - alpha) + alpha * stretch
        #Xc[F_NATTS_IX]  = Xc[F_NATTS_IX]    * (1 - alpha) + alpha * n_atts
        #Xc[FAIL_IX]     += 1.0
        Xc[RECENCY_IX]  = -1.0
    return X