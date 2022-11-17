#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:09:32 2022

"""

import sys
import os 
import math 

#exp for running 20% val to get the overfitting-point 

classOC = int(sys.argv[1])


L = [47,38,35,74,21,35,88,34,36,33]


num_layers = 1
num_heads = 2

length = L[classOC]


num_train = math.floor(length*0.8)
command = "python3 custom_multihead_attention.py " + str(classOC) + " "+ str(num_train) + " " + str(num_layers) + " " + str(num_heads)
#print(command)
os.system(command)
            
        
