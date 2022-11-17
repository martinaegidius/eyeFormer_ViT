#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:20:11 2022

@author: max
"""

import os 
import matplotlib.pyplot as plt

PATH = os.path.dirname(__file__)
if not os.path.exists(PATH+"/outputData/"):
    os.mkdir(PATH+"/outputData/")
    print("Made dir.")


epochPoints = [x for x in range(9)]
lossPoints = [x for x in range(9,18)]

fig = plt.figure(1)
plt.plot(epochPoints,lossPoints)
plt.title("Cross-entropy loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig(str(PATH+"/outputData/loss_graph.png"))

