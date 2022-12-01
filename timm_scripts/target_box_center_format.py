#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:35:07 2022

@author: max
"""

import torch 

sample = torch.tensor([[20,20,100,50],
        [154., 265., 474., 375.]])

sample2 = torch.zeros(2,1,4)
sample2[:,0,:] = sample[:]

def get_center(targets):
    """
    Gets center x, y and bbox w, h from a batched target sequence
    Parameters
    ----------
    targets : batched torch tensor
        contains batched target x0,y0,x1,y1.

    Returns
    -------
    xywh : torch tensor
        bbox cx,cy,w,h

    """
    xywh = torch.zeros(targets.shape[0],1,4,dtype=torch.int32)
    #w = torch.zeros(targets.shape[0],dtype=torch.int32)
    #h = torch.zeros(targets.shape[0],dtype=torch.int32)
    
    xywh[:,0,0]  = torch.div(targets[:,0,2]+targets[:,0,0],2)
    xywh[:,0,1] = torch.div(targets[:,0,-1]+targets[:,0,1],2)
    xywh[:,0,2] = targets[:,0,2]-targets[:,0,0]
    xywh[:,0,3] = targets[:,0,-1]-targets[:,0,1]
    
    return xywh


def center_to_box(targets):
    """
    Scales center-prediction cx,cy,w,h back to x0,y0,y1,y2

    Parameters
    ----------
    targets : batched torch tensor
        Format center x, center y, width and height of box

    Returns
    -------
    box_t : batched torch tensor
        Format lower left corner, upper right corner, [x0,y0,x1,y1]
        
    """
    box_t = torch.zeros(targets.shape[0],1,4,dtype=torch.int32)
    box_t[:,0,0] = targets[:,0,0]-torch.div(targets[:,0,2],2) #x0
    box_t[:,0,1] = targets[:,0,1]-torch.div(targets[:,0,3],2) #y0
    box_t[:,0,2] = targets[:,0,0]+torch.div(targets[:,0,2],2) #x1
    box_t[:,0,3] = targets[:,0,1]+torch.div(targets[:,0,3],2) #y2
    return box_t

cc = get_center(sample2)
cc2 = center_to_box(cc)