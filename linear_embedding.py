#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:56:21 2022

@author: max
"""

import torch 
import torch.nn as nn 


x = torch.rand(4,32,2)

#make a linear embedding 
xF = x.reshape(4,32*2)
embedder = nn.Linear(64,46)
embedder(x)