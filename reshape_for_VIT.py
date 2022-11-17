#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:06:19 2022

@author: max
"""

from load_POET import pascalET
import cv2
import matplotlib.pyplot as plt 
import os
import pickle 
from tqdm import tqdm 

def resize_images_struct(data: pascalET,goal: int,classN: str, WRITE: bool):
    """Takes whole pascalET-dataset-instance and resizes"""
    scale_factors = [] #width,height list
    for k in range(len(data.etData)):
        for i,size in tqdm(enumerate(data.im_dims)):
            #print("Iteration no: ",i)
            h_factor = size[0]/goal #DATA IS IN HEIGHT,WIDTH
            w_factor = size[1]/goal
            scale_factors.append([w_factor,h_factor])
            #print(data.filename[i])
            #print("Resizing factors: ", w_factor,h_factor)
            
            im_path = data.p + "/Data/POETdataset/PascalImages/" +data.classes[k]+"_"+data.filename[i]+".jpg"
            #img = cv2.imread(im_path)
            #resized = cv2.resize(img,(goal,goal))
            
            
            #resized = resized[:,:,::-1] #return in rgb instead of bgr
            
            #now we need eyetracking scaled aswell 
            #may later be implemented as standalone function
            for j in range(data.NUM_TRACKERS):
                data.etData[k][i][:,0][j][:,0]  /= w_factor
                data.eyeData[i][:,0][j][:,1] /= h_factor
            
            #and finally bounding-box-scaling - MAYBE IMPLEMENT FUNCTION
            x,y,w,h = data.get_bounding_box(data.bbox[0])
            w = (x+w)/w_factor
            h = (y+h)/h_factor
            x /= w_factor
            y /= h_factor
            
            if data.bbox[i].ndim>1:
                data.bbox[i][0][0] = int(x)
                data.bbox[i][0][1] = int(y)
                data.bbox[i][0][2] = int(w)
                data.bbox[i][0][3] = int(h)
                #alternatively data.bbox[i][0] = np.array([x,y,w,h])
            else: 
                data.bbox[i][0] = int(x)
                data.bbox[i][1] = int(y)
                data.bbox[i][2] = int(w)
                data.bbox[i][3] = int(h)
                
            tmp_filename = res_path+data.chosen_class+"_"+data.filename[i]
            #print("Trying to write to: ",tmp_filename)
            if WRITE==True:
                cv2.imwrite(tmp_filename,resized)
            
            #link path to /resized/filename.jpg instead in Data-datastructure - may actually be overly complicated for no reason
            #fixed with resized=True flag in plot-function etc
            #data.filename[i] = "Resized/"+data.filename[i]
            #print("relinking filepath to: ",data.filename[i])
            
        return resized

def resize_images_for_class(data: pascalET,goal: int,classN: str):
    scale_factors = [] #width,height list
    for i,size in tqdm(enumerate(data.im_dims)):
        #print("Iteration no: ",i)
        h_factor = size[0]/goal #DATA IS IN HEIGHT,WIDTH
        w_factor = size[1]/goal
        scale_factors.append([w_factor,h_factor])
        #print(data.filename[i])
        #print("Resizing factors: ", w_factor,h_factor)
        im_path = data.p + "/Data/POETdataset/PascalImages/" +classN+"_"+data.filename[i]
        img = cv2.imread(im_path)
        resized = cv2.resize(img,(goal,goal))
        
        
        #resized = resized[:,:,::-1] #return in rgb instead of bgr
        
        #now we need eyetracking scaled aswell 
        #may later be implemented as standalone function
        for j in range(data.NUM_TRACKERS):
            data.eyeData[i][:,0][j][:,0]  /= w_factor
            data.eyeData[i][:,0][j][:,1] /= h_factor
        
        #and finally bounding-box-scaling - MAYBE IMPLEMENT FUNCTION
        x,y,w,h = data.get_bounding_box(data.bbox[0])
        w = (x+w)/w_factor
        h = (y+h)/h_factor
        x /= w_factor
        y /= h_factor
        
        if data.bbox[i].ndim>1:
            data.bbox[i][0][0] = int(x)
            data.bbox[i][0][1] = int(y)
            data.bbox[i][0][2] = int(w)
            data.bbox[i][0][3] = int(h)
            #alternatively data.bbox[i][0] = np.array([x,y,w,h])
        else: 
            data.bbox[i][0] = int(x)
            data.bbox[i][1] = int(y)
            data.bbox[i][2] = int(w)
            data.bbox[i][3] = int(h)
            
        tmp_filename = res_path+data.chosen_class+"_"+data.filename[i]
        #print("Trying to write to: ",tmp_filename)
        cv2.imwrite(tmp_filename,resized)
        
        #link path to /resized/filename.jpg instead in Data-datastructure - may actually be overly complicated for no reason
        #fixed with resized=True flag in plot-function etc
        #data.filename[i] = "Resized/"+data.filename[i]
        #print("relinking filepath to: ",data.filename[i])
        
    return resized

DEBUG = False

dset = pascalET() #init
dset.loadmat()#specific([0,8]) #load airplanes, motorbikes - the easiest data-points
dset.load_images_for_class(8) #load image-files for one class
#dset.random_sample_plot()

res_path = dset.p+"/Data/POETdataset/PascalImages/Resized/"

if not os.path.exists(res_path):
    os.mkdir(res_path)
    print("Created dir in: ",res_path)

if DEBUG==True:
    dset.specific_plot(8,"2009_004207.jpg") #plot original for debugging


img = resize_images_for_class(dset,goal=224,classN="motorbike")


file_resized = open(res_path+"resized_pascalET.obj","wb")
#save as a small pickle
pickle.dump(dset,file_resized)

#PROGRESS: you now have motorbikes. Ideally you would like motorbikes and aeroplanes, so you should write a data-loader function which can handle two classes.


if DEBUG==True:
    ax = plt.gca()
    plt.figure(1)
    
    #plotting
    x,y,w,h = dset.get_bounding_box(dset.bbox[0])
    from matplotlib.patches import Rectangle
    
    ax = plt.gca()
    rect = Rectangle((x,y),w,h,linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    for i in range(dset.NUM_TRACKERS):
        color = next(ax._get_lines.prop_cycler)['color']
        mylabel = str(i+1)
        num_fix = int(dset.eyeData[0,i][0][:,0].shape[0]/2)
        print(num_fix) #number of fixations on img
        #Left eye
        plt.scatter(dset.eyeData[0,i][0][:,0],dset.eyeData[0,i][0][:,1],alpha=0.8,label=mylabel,color = color) #wrong - you are not plotting pairwise here, you are plotting R as function of L 
        plt.plot(dset.eyeData[0,i][0][0:num_fix,0],dset.eyeData[0,i][0][0:num_fix,1],label=str(),color= color)
        plt.plot(dset.eyeData[0,i][0][num_fix:,0],dset.eyeData[0,i][0][num_fix:,1],label=str(),color = color)
    plt.legend()
    plt.imshow(img)
    
    
    #filename[0] = 2010_003635.jpg    
    #dset.eyeData[0]