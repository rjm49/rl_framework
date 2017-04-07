'''
Created on 6 Mar 2017

@author: Russell
'''
import codecs
from matplotlib import pyplot
from reinf.exp1.students.forgetting import ForgettingStudent, ForgetMode
from scipy.optimize.minpack import curve_fit
from numpy import exp, power, log, math, average



def get_retention(t):
#     c = 1.8
#     d = 1.21
    c = average([1.8, 1.34, 0.9, 1.36]) 
    d = average([1.21, 0.873, 0.9, 1.36])
    print(c,d)
    if t<=1.0:
        return 1.0
    innr = math.pow( math.log(t, 10.0), d)
    return c / (innr + c)  


def fit_wickelgren(t, lamb,beta,psi):
    return lamb * power(1.0 + beta*t, psi)

def fit_ebb_power(t,a,b):
    eps = 0.0
    ep_innr = 1.0 - power((2.0/(t+eps)), a)
    return 1.0 - power(ep_innr, b)

def fit_expon(t, b, m):
    return b / exp(m*t)

def fit_power(t, b,m):
    eps = 0.0
    return b / power(t,m)

if __name__ == '__main__':


    pyplot.xlabel("# Min")
    pyplot.ylabel("Retention")

    steps = []
    m_exp = []
    m_ebp = []
    m_log = []
    m_wic = []
    
    steps = [20,60,9*60,24*60, 2*24*60] #, 6*24*60, 31*24*60]
        
    for t_min in steps:
        m_log.append( get_retention(t_min))

    print(steps)
    print(m_log)
    pyplot.plot(steps, m_log, label="log'm")
    
#     wparams, wcov = curve_fit(fit_wickelgren, steps, m_log)
    pparams, pcov = curve_fit(fit_ebb_power, steps, m_log)
    xparams, xcov = curve_fit(fit_expon, steps, m_log)
    oparams, wcov = curve_fit(fit_power, steps, m_log)

    
    m_exp = []
    m_pow = []
    
    steps1 = range(1,2880,10)
    for t in steps1:
        m_ebp.append( fit_ebb_power(t, *pparams) )
#         m_wic.append( fit_wickelgren(t, *wparams))
        m_exp.append( fit_expon(t, *xparams))
        m_pow.append( fit_power(t, *oparams))

    print(oparams)
    
    pyplot.plot(steps1, m_ebp, label="ebb_p")
#     pyplot.plot(steps1, m_wic, label="wickl")
    pyplot.plot(steps1, m_exp, label="expon")
    pyplot.plot(steps1, m_pow, label="power")
    
    leg = pyplot.legend(loc='upper right')
    leg.get_frame().set_alpha(0.3)
    pyplot.show()