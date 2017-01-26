'''
Created on 20 Jan 2017

@author: Russell
'''
import time
import sys

if __name__ == '__main__':

    for i in range(100):
        time.sleep(1)
        sys.stdout.write("\r%d%%" % i)
        sys.stdout.flush()
        #print("\r{}%".format(i), end='') #PY3 version
        #print "\r{}%".format(i), #PY2 version