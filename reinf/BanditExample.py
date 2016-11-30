'''
Created on 14 Oct 2016

@author: Russell
'''
from reinf.NarmedBanditTestbed import NarmedBanditTestbed
import random
from test.test_math import acc_check
from random import choice

if __name__ == '__main__':
    bandit = NarmedBanditTestbed()
    n = bandit.n
    T = 100000
    optimistic = False
    greedy=False
    opt = 10
    estimated_vals = [0 for i in range(n)] if not optimistic else [opt for i in range(n)]
    learn_rate = 10.0
    acc_reward = 0
    a = None
    
    for k in range(1,T+1):

        max_value = max(estimated_vals)        
        if (not greedy and 0 == k % 10): #implement a 1 in x exploratory step
            print("randomise!")
            non_max_indices = [i for i, x in enumerate(estimated_vals) if x!=max_value]
            a  = choice(non_max_indices) #make a non-greedy choice to explore
            #a = random.randint(0,n-1)
        else:
            max_indices = [i for i, x in enumerate(estimated_vals) if x==max_value]
            a  = choice(max_indices) #make a random choice from the maxima
        
        rk = bandit.select_action(a)
        print(k,":", rk, "from action", a)
        qk = estimated_vals[a]
        
        qk1 = qk + learn_rate*(rk - qk)/k
#        print ("est val",a,"=",qk1)
        estimated_vals[a] = qk1

        acc_reward += rk
    
    print(estimated_vals)
    print(bandit.means)
    print(acc_reward)
    
    print("max estimated lever is : ", estimated_vals.index(max(estimated_vals)), "with value:", max(estimated_vals))
    