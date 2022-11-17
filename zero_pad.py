#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:46:49 2022

@author: max
"""

import numpy as np
import torch

a = np.array([[186.9,  83. ],
       [295.3, 145.1],
       [215.4, 180.7],
       [164.5, 178.4],
       [220. , 178.7],
       [156.1, 159.4],
       [182.1,  81. ],
       [284.2, 147.7],
       [202.1, 177.6],
       [168.4, 168.2],
       [238.3, 164.6],
       [177.5, 134.7]])



def zero_pad(inArr: np.array,padto: int,padding: int):
    if(inArr.shape[0]>=padto):
        return torch.from_numpy(inArr[:padto]).type(torch.float32)
    
    else: 
        outTensor = torch.zeros((padto,2)).type(torch.float32) #[32,2]
        numEntries = inArr.shape[0]
        outTensor[:numEntries,:] = torch.from_numpy(inArr[-numEntries:,:]) #get all
        return outTensor


print(zero_pad(a,32,0))


b = np.random.rand(34,2)
c = np.random.rand(32,2)
print(zero_pad(b,32,0))
print(zero_pad(c,32,0))
