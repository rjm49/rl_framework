'''
Created on 14 Oct 2016

@author: Russell
'''
import math
import random

class NarmedBanditTestbed(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.T = 1000 # number of steps in the test run
        self.n = 10 #number of arms the bandit has
        self.means = []
        self.sdevs = []
        self.reward_funcs = []
        
        #set up the gaussian reward functions (these are random)
        for _ in range(self.n):
            mu = random.randint(-5,5)
            sigma = random.uniform(0,1)
            self.means.append(mu)
            self.sdevs.append(sigma)
            
            
    def select_action(self, a):
        mu = self.means[a]
        sigma = self.sdevs[a]
        return random.gauss(mu, sigma) #this is the reward we get from having selected action a
        
    