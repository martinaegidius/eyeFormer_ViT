#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 11:32:42 2022

@author: max
"""

import os 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import rc,rcParams

#from matplotlib.ticker import FormatStrFormatter

path1 = os.path.dirname(__file__) + "/Results/"
path = path1 + "1st_exp_res_fixed_val_delim.csv"
data = pd.read_csv(path)


sns.set()

# #sns.set_theme(style="whitegrid")
# #norm = plt.Normalize(data.N_L.min(),data.N_L.max())
# ax = sns.relplot(x=data.L,y=data.Tr_loc,hue=data.N_L,size=data.Tr_acc,sizes=(40,400),alpha=0.8,palette="flare")
# #sm = plt.cm.ScalarMappable(cmap="RdBu",norm=norm)
# #sm.set_array([])

# #ax.get_legend().remove()
# #ax.figure.colorbar(sm)
# plt.show()


# ax = sns.relplot(x=data.L,y=data.Te_loc,hue=data.N_L,size=data.Te_acc,sizes=(40,400),alpha=0.8,palette="flare")
# #sm = plt.cm.ScalarMappable(cmap="RdBu",norm=norm)
# #sm.set_array([])

# #ax.get_legend().remove()
# #ax.figure.colorbar(sm)
# plt.show()


"""

plt.figure(1)
ax = sns.scatterplot(x=data.Te_loc,y=data.Tr_loc,hue=data.N_L,alpha=0.8,size=data.L,sizes=(80,400),palette="deep")
ax.set_xlabel("Test CorLoc")
ax.set_ylabel("Train CorLoc")
plt.show()

plt.figure(2)
ax = sns.scatterplot(x=data.Te_loc,y=data.Tr_loc,hue=data.N_H,alpha=0.8,size=data.L,sizes=(80,400),palette="deep")
ax.set_xlabel("Test CorLoc")
ax.set_ylabel("Train CorLoc")
plt.show()

plt.figure(3)
ax = sns.scatterplot(x=data.Te_loc,y=data.Tr_loc,hue=data.N_L,alpha=0.8,size=data.N_H,sizes=(80,400),palette="deep")
ax.set_xlabel("Test CorLoc")
ax.set_ylabel("Train CorLoc")
plt.show()"""

rc('font', weight='bold')
#make equispaced x-axis
x = data.L.to_numpy(dtype=np.float32)
x[x==1] = 0
x[x==4] = 1
x[x==8] = 2
x[x==16] = 3
#x[x==32] = 4

noise = np.random.normal(0,0.07,len(x))
#noiseY = np.random.normal(0,0.01,len(x))
x += noise


"""FOR DUAL"""
fig,axes = plt.subplots(2,1, sharex=True,figsize=(13,11))
plt.subplots_adjust(hspace=0.08)

g = sns.scatterplot(ax=axes[0],x=x,y=data.Tr_loc,hue=data.N_L,style=data.N_H,alpha=0.6,palette="deep",s=150,markers=["o","D"])
axes[0].set_xlabel("Training-set length",fontsize=20,fontweight="bold")
axes[0].set_ylabel("Train CorLoc",fontsize=20,fontweight="bold")
axes[0].set_xticks(np.arange(4))
#axes[0].set_yticks(np.arange(0,1.2,0.2))
axes[0].set_xticklabels(["1","4","8","16"])
#axes[0].set_yticklabels(["0.0","0.2","0.4","0.6","0.8","1.0"])
axes[0].tick_params(axis='both', which='major', labelsize=20)
#axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#plt.legend(loc="upper center",ncol=8)
#handles = g._legend_data.values()
#labels = g._legend_data.keys()



h = sns.scatterplot(ax=axes[1],x=x,y=data.Te_loc,hue=data.N_L,style=data.N_H,alpha=0.6,palette="deep",s=150,markers=["o","D"])
h.legend_.remove()
axes[1].set_xlabel("Training-set length",fontweight="bold")
axes[1].set_ylabel("Test CorLoc")
axes[1].set_xticks(np.arange(4))
axes[1].set_xticklabels(["1","4","8","16"])
axes[1].tick_params(axis='both', which='major', labelsize=20)
axes[1].set_ylabel("Test CorLoc",fontsize=20,fontweight="bold")
#make row legend
sns.move_legend(g,"upper center",bbox_to_anchor=(0.5,1.2),ncol=8)
for lh in axes[0].legend_.legendHandles: 
    lh.set_alpha(0.7)
    lh._sizes = [70] 

plt.show()
plt.savefig(path1+"both_results_on_experiment1.pdf")


"""FOR SINGLE"""
path1 = os.path.dirname(__file__) + "/Results/"
path = path1 + "1st_exp_res_fixed_val_delim.csv"
data = pd.read_csv(path)


sns.set()
rc('font', weight='bold')
#make equispaced x-axis
x = data.L.to_numpy(dtype=np.float32)
x[x==1] = 0
x[x==4] = 1
x[x==8] = 2
x[x==16] = 3
#x[x==32] = 4

noise = np.random.normal(0,0.07,len(x))
#noiseY = np.random.normal(0,0.01,len(x))
x += noise


fig,axes = plt.subplots(1,1, sharex=True,figsize=(16,8))
plt.subplots_adjust(hspace=0.08)

g = sns.scatterplot(ax=axes,x=x,y=data.Tr_loc,hue=data.N_L,style=data.N_H,alpha=0.6,palette="deep",s=150,markers=["o","D"])
axes.set_xlabel("Training-set length",fontsize=20,fontweight="bold")
axes.set_ylabel("Train CorLoc",fontsize=20,fontweight="bold")
axes.set_xticks(np.arange(4))
#axes[0].set_yticks(np.arange(0,1.2,0.2))
axes.set_xticklabels(["1","4","8","16"])
#axes[0].set_yticklabels(["0.0","0.2","0.4","0.6","0.8","1.0"])
axes.tick_params(axis='both', which='major', labelsize=20)
#axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#plt.legend(loc="upper center",ncol=8)
#handles = g._legend_data.values()
#labels = g._legend_data.keys()
axes.set_xlabel("Training-set length",fontweight="bold")
for lh in g.legend_.legendHandles: 
    lh.set_alpha(0.7)
    lh._sizes = [70] 
    # You can also use lh.set_sizes([50])
sns.move_legend(g,"upper center",bbox_to_anchor=(0.5,1.13),fontsize=18,ncol=8)
#g.legend(fontsize=14)

plt.savefig(path1+"results_on_experiment1.pdf")




