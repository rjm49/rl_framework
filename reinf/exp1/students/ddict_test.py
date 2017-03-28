'''
Created on 24 Mar 2017

@author: Russell
'''
from _collections import defaultdict

dd = defaultdict(int)
x = dd[42]
print(x)
for y in dd:
    print(y)