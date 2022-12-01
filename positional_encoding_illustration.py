#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:07:27 2022

@author: max
"""

import torch 
import math 

def efficientPosEnc(seq_len,d_model,n):
    """Generate positional encoding table for input specified"""
    pe_matrix = torch.zeros(seq_len,d_model)
    pe_matrix = torch.zeros(seq_len,d_model)
    position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) #unsqueeze to make it a column-vector
    posIdx = torch.arange(0,d_model,2).float() #(start,end,step). Possible values of i
    denominator = torch.exp(posIdx*(-math.log(10000))/d_model) #is 1/y^(x/d_model)
    pe_matrix[:,0::2] = torch.sin(position*denominator)
    pe_matrix[:,1::2] = torch.cos(position*denominator)
    return pe_matrix


def getPosEnc(seq_len=32,d_model=4,n=10000):
    """Illustrative but computationally inefficient method for generating positional encoding"""
    P = torch.zeros((seq_len,d_model))
    for pos in range(seq_len): #rows 
        for i in torch.arange(int(d_model/2)): #columns
            denominator = torch.pow(n,2*i/d_model)
            P[pos,2*i] = torch.sin(pos/denominator) #even entries
            P[pos,2*i+1] = torch.cos(pos/denominator)
    return P 

P = getPosEnc()


def illustratePos(matrix):
    
    import matplotlib.pyplot as plt
    #import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #mpl.rcParams['axes.labelsize'] = 17
    #mpl.rcParams['xtick.labelsize'] = 17
    #mpl.rcParams['ytick.labelsize'] = 17
    #mpl.rcParams['legend.fontsize'] = 10
    fig, ax = plt.subplots(1,1,figsize=(17,9),dpi=120)
    ax.grid(None)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im = ax.imshow(matrix,interpolation='nearest', aspect='auto',cmap="viridis")
    cbar = fig.colorbar(im,cax=cax,orientation='vertical')
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(17)
    ax.set_ylabel("Position in sequence [idx]",fontsize=23,fontweight="bold")#,fontweight="bold")
    ax.set_xlabel("Embedded feature depth",fontsize=23,fontweight="bold")#,fontweight="bold")
    ax.xaxis.set_label_position('bottom') 
    ax.tick_params(axis='both', which='major', labelsize=19)#,width=2)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
        
    fig.show()
    plt.grid(False)
    fig.tight_layout()
    fig.savefig("positional_encoding.pdf",dpi=fig.dpi)
    return None

#PE = efficientPosEnc(32,46,10000)
PE = getPosEnc(32,50,10000)
illustratePos(PE)
    