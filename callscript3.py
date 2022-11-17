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
EPOCHS = [485,139,787,123,609,329,497,451,137,115]

num_layers = 6
num_heads = 2
EVAL = 1

length = L[classOC]

command = "python3 custom_multihead_attention.py " + str(classOC) + " "+ str(length) + " " + str(num_layers) + " " + str(num_heads) + " " + str(EPOCHS[classOC]) + " " + str(EVAL)
#print(command)
os.system(command)
            
        
