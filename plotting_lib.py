#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:51:26 2022

@author: max
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch 
import os 
from random import sample
import math

#-------------------Script flags----------------------#
DEBUG = True

#-----------------------END>--------------------------#
className = "diningtable"
impath = os.path.dirname(__file__)+"/Data/POETdataset/" #path for image-data
dpath = impath + className #path for saved model-results
impath += "/PascalImages/"


def load_results(root_dir,className,file_specifier):
    path = root_dir + "/" + className+"_"+file_specifier+".pth"
    if(DEBUG):
        print("path is: ",path)
    try: 
        entrys = torch.load(path)
        print("...Success loading...")
        return entrys
    except: 
        print("Error loading file. Check filename/path: ",path)
        return None

def rescale_coords(data):
    #takes imdims and bbox data and rescales. [MUTATES]
    #Returns tuple of two tensors (target,preds)
    imdims = data[-1].squeeze(0).clone().detach()
    target = data[3].squeeze(0).clone().detach()
    preds = data[4].squeeze(0).clone().detach()
    
    target[0::2] = target[0::2]*imdims[-1]
    target[1::2] = target[1::2]*imdims[0]
    preds[0::2] = preds[0::2]*imdims[-1]
    preds[1::2] = preds[1::2]*imdims[0]
    
    #print("Rescaled target: ",target)
    #print("Rescaled preds: ",preds)
    
    return target,preds
    
def single_rescale_coords(box,data):
    #takes imdims from data and bbox data and rescales. Return type: tensor with scaled box coordinates, fitting on image-dims
    imdims = data[-1].squeeze(0).clone().detach()
    nbox = torch.zeros(4)
    nbox[0::2] = box[0::2]*imdims[-1]
    nbox[1::2] = box[1::2]*imdims[0]
    return nbox

def single_format_bbox(box,data):
    """Returns bounding box in format ((x0,y0),w,h) for a mean-model-box
    Args: 
        input: 
            box: generated mean-data-box
            data: saved list-of-list-of-list containing model-data and image-names. 
                Format: [filename,class,IOU,target,output,size]
        returns: 
            Bounding box coordinates in format ((x0,y0),w,h) for mean/median box
    """   
    scaled = single_rescale_coords(box,data)
    x0 = scaled[0]
    y0 = scaled[1]
    w = scaled[2] - x0
    h = scaled[3] - y0
    #print("x0: ",x0,"y0: ",y0,"w: ",w,"h: ",h)
    return [x0,y0,w,h]

def bbox_for_plot(data):
    """Returns bounding box in format ((x0,y0),w,h)
    Args: 
        input: 
            data: saved list-of-list-of-list containing model-data and image-names. 
                Format: [filename,prediction-result,IOU,target,output,size]
        returns: 
            Bounding box coordinates in format ((x0,y0),w,h) for preds and target 
            (target,preds)
    """   
    targets,preds = rescale_coords(data)
    
    #for target
    x0 = targets[0]
    y0 = targets[1]
    w = targets[2] - x0
    h = targets[3] - y0
    #print("x0: ",x0,"y0: ",y0,"w: ",w,"h: ",h)
    target = [x0,y0,w,h]
    
    #for preds 
    x0 = preds[0]
    y0 = preds[1]
    w = preds[2] - x0
    h = preds[3] - y0
    #print("x0: ",x0,"y0: ",y0,"w: ",w,"h: ",h)
    preds = [x0,y0,w,h]
    
    return target,preds
    
def mean_model(data): #does not work for batched data yet. 
    #Generate mean-model of training-data
    holder_t = torch.zeros(len(data),4)
    for entry in range(len(data)):
        #print(data[entry][3].squeeze(0))
        holder_t[entry] = data[entry][3].squeeze(0)
    return torch.mean(holder_t,0)
        
def median_model(data): #does not work for batched data yet.
    #Generate median-model of training-data
    holder_t = torch.zeros(len(data),4)
    for entry in range(len(data)):
        #print(data[entry][3].squeeze(0))
        holder_t[entry] = data[entry][3].squeeze(0)
    return torch.median(holder_t,0)[0]
    
    

def plot_train_set(data,root_dir,classOC=0):
    """Prints subset of data which is overfit upon
    Args: 
        input: 
            data: saved list-of-list-of-list containing model-data and image-names
                Format: [filename,class,IOU,target,output,size]
            root_dir: imagedir
            classOC: descriptor of class which is used
        returns: 
            None
    """                  
    meanBox = mean_model(data)
    medianBox = median_model(data)          
    NSAMPLES = len(data)
    NCOLS = 2
    
    if NSAMPLES > 9: 
        NSAMPLES = 9
        data = sample(data,9)
        NCOLS = 3
    
    fig, ax = plt.subplots(math.ceil(NSAMPLES/NCOLS),NCOLS)
    col = 0
    row = 0
    for entry in range(NSAMPLES):
        filename = data[entry][0][0]
        im = plt.imread(root_dir+filename)
        target,preds = bbox_for_plot(data[entry])
        mean_box = single_format_bbox(meanBox, data[entry])
        median_box = single_format_bbox(medianBox,data[entry])
        rectT = patches.Rectangle((target[0],target[1]), target[2], target[3], linewidth=2, edgecolor='r', facecolor='none')
        rectP = patches.Rectangle((preds[0],preds[1]), preds[2], preds[3], linewidth=2, edgecolor='m', facecolor='none')
        rectM = patches.Rectangle((mean_box[0],mean_box[1]),mean_box[2],mean_box[3],linewidth=1,edgecolor='b',facecolor='none')
        rectMed = patches.Rectangle((median_box[0],median_box[1]),median_box[2],median_box[3],linewidth=1,edgecolor='lightcoral',facecolor='none')
        if entry%NCOLS==0 and entry!=0: #NUMBER OF COLS
            row += 1 
            col = 0 
        print("Col: ",col,"Row:",row)
        ax[row,col].imshow(im)
        ax[row,col].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
        ax[row,col].add_patch(rectT)
        ax[row,col].add_patch(rectP)
        ax[row,col].add_patch(rectM)
        ax[row,col].add_patch(rectMed)
        
        
        #plt.text(target[0]+target[2]-10,target[1]+10, "IOU", bbox=dict(facecolor='red', alpha=0.5))
        #ax[row,col].text(data[entry][4][0][2].item()-0.1,data[entry][4][0][1].item()+0.1,"IOU:{:.2f}".format(data[entry][2][0].item()),bbox=dict(facecolor='magenta', alpha=0.5),transform=ax[row,col].transAxes)
        ax[row,col].text(0.02,0.05,"IOU:{:.2f}".format(data[entry][2]),bbox=dict(facecolor='magenta', alpha=0.75),transform=ax[row,col].transAxes)
        ax[row,col].set_title(filename,fontweight="bold",size=8)
        col += 1
    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.handleheight'] = 1.125
    fig.legend((rectT,rectP,rectM,rectMed),("Target","Prediction","Mean box","Median box"),loc="upper center",ncol=4,framealpha=0.0,bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Predictions on subset of trainset")
    plt.show()
    return None



            
def plot_test_set(data,root_dir,classOC=0,meanBox=None,medianBox=None,mode=None):
    """Prints subset of data which model is tested on
    Args: 
        input: 
            data: saved list-of-list-of-list containing model-data and image-names
                Format: [0: filename,
                         1: prediction-result (0:false,1:correct),
                         2: IOU-score
                         4: target [tensor]
                         5: prediction [tensor],
                         6: image-dimensions in h,w]
            root_dir: imagedir
            classOC: descriptor of class which is used
            meanBox: box with means of trainset
            medianBox: box with medians of trainset
        returns: 
            None
    """      
            
    NSAMPLES = len(data)
    NCOLS = 2
    if NSAMPLES > 9: 
        NSAMPLES = 9
        if(mode!=None):
            tmpData = []
            for i in range(len(data)):
                if(mode=="success"):
                    if(int(data[i][1])==1):
                        tmpData.append(data[i])
                elif(mode=="failure"):
                    if(int(data[i][1])==0):
                        tmpData.append(data[i])
            data = tmpData
            del tmpData
            
        
    data = sample(data,NSAMPLES)
    NCOLS = 3
    
    fig, ax = plt.subplots(math.ceil(NSAMPLES/NCOLS),NCOLS)
    col = 0
    row = 0
    for entry in range(NSAMPLES):
        filename = data[entry][0][0]
        im = plt.imread(root_dir+filename)
        target,preds = bbox_for_plot(data[entry])
        if(meanBox!=None):
            mean_box = single_format_bbox(meanBox, data[entry])
            rectM = patches.Rectangle((mean_box[0],mean_box[1]),mean_box[2],mean_box[3],linewidth=1,edgecolor='b',facecolor='none')
        if(medianBox!=None):
            median_box = single_format_bbox(medianBox,data[entry])
            rectMed = patches.Rectangle((median_box[0],median_box[1]),median_box[2],median_box[3],linewidth=1,edgecolor='lightcoral',facecolor='none')
        rectT = patches.Rectangle((target[0],target[1]), target[2], target[3], linewidth=2, edgecolor='r', facecolor='none')
        rectP = patches.Rectangle((preds[0],preds[1]), preds[2], preds[3], linewidth=2, edgecolor='m', facecolor='none')
        
        if entry%NCOLS==0 and entry!=0: #NUMBER OF COLS
            row += 1 
            col = 0 
        print("Col: ",col,"Row:",row)
        ax[row,col].imshow(im)
        ax[row,col].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
        ax[row,col].add_patch(rectT)
        ax[row,col].add_patch(rectP)
        legend_tuple = (rectT,rectP)
        legend_name_tuple = ("Target","Prediction")
        legend_ncol = 2
        
        if(meanBox!=None):
            ax[row,col].add_patch(rectM)
            new_legend_tuple = legend_tuple + (rectM,)
            new_legend_name_tuple = legend_name_tuple + ("Mean box",)
            legend_tuple = new_legend_tuple 
            legend_name_tuple = new_legend_name_tuple
            legend_ncol += 1
            del new_legend_tuple,new_legend_name_tuple
            
        if(medianBox!=None):
            ax[row,col].add_patch(rectMed)
            new_legend_tuple = legend_tuple + (rectMed,)
            new_legend_name_tuple = legend_name_tuple + ("Median box",)
            legend_tuple = new_legend_tuple 
            legend_name_tuple = new_legend_name_tuple
            legend_ncol += 1
            del new_legend_tuple,new_legend_name_tuple
            
        
        ax[row,col].text(0.02,0.05,"IOU:{:.2f}".format(data[entry][2]),bbox=dict(facecolor='magenta', alpha=0.75),transform=ax[row,col].transAxes)
        ax[row,col].set_title(filename,fontweight="bold",size=8)
        #ax[row,col].text(preds[2],preds[3],"{:.2f}".format(data[entry][2][0].item()),bbox=dict(facecolor='magenta', alpha=0.5),transform=ax[row,col].transAxes)
        col += 1
    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.handleheight'] = 1.125
    fig.legend(legend_tuple,legend_name_tuple,loc="upper center",ncol=legend_ncol,framealpha=0.0,bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Predictions on subset of testset with mode: "+mode)
    plt.show()
    return None
    


        

    
testdata = load_results(dpath,className,"test_on_test_results")
testOnTrain = load_results(dpath,className,"test_on_train_results")
plot_train_set(testOnTrain,impath)

mean = mean_model(testOnTrain)
median = median_model(testOnTrain)
plot_test_set(testdata,impath,classOC=5,meanBox=mean,medianBox=median,mode="success")
plot_test_set(testdata,impath,classOC=5,meanBox=mean,medianBox=median,mode="failure")



mean = mean_model(testOnTrain)
median = median_model(testOnTrain)



       
      