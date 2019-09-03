# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:14:26 2019

@author: hyunj
"""


import os


i = 0
j = 3

for filename in os.listdir("."):
    
    k = i % j + 1
    
    l = divmod(i, j)[0] + 1

#    os.listdir(".")[0][:-25] + ' ' + str(k) + ' ' + str(l) +'.txt'
    
    os.rename(filename,    str(l) + ' ' + filename[:-25] + ' ' + str(k) + '.txt')
        
    i = i + 1
    
    print(str(k) + ' ' + str(l))
    
    if i == 60:
        break