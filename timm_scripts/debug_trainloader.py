#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 10:56:20 2022

@author: max
"""

import torch
import math
import os 
import numpy as np 
import matplotlib.pyplot as plt
import torchvision.transforms 
import torchvision.io
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import PIL.Image as Image
import timm
import matplotlib.pyplot as plt
import matplotlib as mpl 
import torch.nn as nn
import ViT_model as vm
from tqdm import tqdm
#import handle_pascal as hdl
from load_POET_timm import pascalET

classesOC = [x for x in range(10)]

def generate_DS(dataFrame,classes=[x for x in range(10)]):
    """ Note: 
        load classes from .mat-format. 
        
        Args
        classes: list of input-classes of interest {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}   
        If none is provided, fetch all 9 classes
        
        returns: 
            A tensor which contains: 1. image-filenames, 2. bounding box, 3. class number, 4:8: a list of processed cleaned both-eye-fixations
            A list of filenames
            A tensor which contains all eyetracking data
            A tensor containing bounding boxes
            A tensor containing numerated classlabels
            A tensor containing image-dimensions
    """
    dataFrame.loadmat()
    dataFrame.convert_eyetracking_data(CLEANUP=True,STATS=True)
    NUMCLASSES = len(classes) #TODO: IMPLEMENT GENERATE_DS FOR MORE THAN TWO CLASSES
    
    
    dslen = 0
    dslen_li = []
    for CN in classes: 
        #print(CN)
        dslen += len(dataFrame.etData[CN])
        dslen_li.append(len(dataFrame.etData[CN]))
        #print("Total length is ",dslen)
        
    #print(dslen_li)
    
    df = np.empty((dslen,8),dtype=object)
    
    #1. fix object type by applying zero-padding
    num_points = 32
    eyes = torch.zeros((dslen,dataFrame.NUM_TRACKERS,num_points,2))
    #2.saving files to list
    filenames = []
    #3: saving target in its own array
    targets = np.empty((dslen,4))
    #classlabels and imdims to own array
    classlabels = np.empty((dslen))
    imdims = np.empty((dslen,2))
    
    
    
    datalist = []
    for CN in classes:
        datalist.append(dataFrame.etData[CN])
        #print("DSET {} LENGTH is: ".format(CN),len(dataFrame.etData[CN]))
    
    idx = 0
    IDXSHIFTER = 0
    k = 0
    for i in range(dslen): 
        if((i==sum(dslen_li[:k+1])) and (i!=0)): #evaluates to true when class-set k is depleted
            #print("depleted dataset {} at idx {}".format(k,i))
            IDXSHIFTER += dslen_li[k]
            k += 1
            
        idx = i - IDXSHIFTER
        df[i,0] = [dataFrame.etData[classes[k]][idx].filename+".jpg"]
        filenames.append(dataFrame.classes[classes[k]]+"_"+dataFrame.etData[classes[k]][idx].filename+".jpg")
        df[i,1] = dataFrame.etData[classes[k]][idx].gtbb
        df[i,2] = classes[k]
        targets[i] =  dataFrame.etData[classes[k]][idx].gtbb
        classlabels[i] = classes[k]
        imdims[i] = dataFrame.etData[classes[k]][idx].dimensions[:2] #some classes also have channel-dim
        for j in range(dataFrame.NUM_TRACKERS):
            sliceShape = dataFrame.eyeData[classes[k]][idx][j][0].shape
            #print("sliceShape becomes: ",sliceShape)
            df[i,3+j] = dataFrame.eyeData[classes[k]][idx][j][0] #list items
            if(sliceShape != (2,0) and sliceShape != (0,) and sliceShape != (2,)): #for all non-empty
                eyes[i,j,:sliceShape[0],:] = torch.from_numpy(dataFrame.eyeData[classes[k]][idx][j][0][-32:].astype(np.int32)) #some entries are in uint16, which torch does not support
            else: 
                eyes[i,j,:,:] = 0.0
                #print("error-filled measurement [entryNo,participantNo]: ",idx,",",j)
                #print(eyes[i,j])
            
            
            
    print("Total length is: ",dslen)
    
    targetsTensor = torch.from_numpy(targets)
    classlabelsTensor = torch.from_numpy(classlabels)
    imdimsTensor = torch.from_numpy(imdims)
    
    return df,filenames,eyes,targetsTensor,classlabelsTensor,imdimsTensor,dslen

def generate_DS_old(dataFrame,classes=[x for x in range(10)]):
    """ Note: as implemented now works for two classes only
    Args
        classes: list of input-classes of interest {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}   
        If none is provided, fetch all 9 classes
        
        returns: 
            A tensor which contains: 1. image-filenames, 2. bounding box, 3. class number, 4:8: a list of processed cleaned both-eye-fixations
            A list of filenames
            A tensor which contains all eyetracking data
            A tensor containing bounding boxes
            A tensor containing numerated classlabels
            A tensor containing image-dimensions
    """
    dataFrame.loadmat()
    dataFrame.convert_eyetracking_data(CLEANUP=True,STATS=True)
    
    dslen = 0
    for CN in classes: 
        dslen += len(dataFrame.etData[CN])
    
    df = np.empty((dslen,8),dtype=object)
    
    #1. fix object type by applying zero-padding
    num_points = 32
    eyes = torch.zeros((dslen,dataFrame.NUM_TRACKERS,num_points,2))
    #2.saving files to list
    filenames = []
    #3: saving target in its own array
    targets = np.empty((dslen,4))
    #classlabels and imdims to own array
    classlabels = np.empty((dslen))
    imdims = np.empty((dslen,2))
    
    
    
    datalist = []
    for CN in classes:
        datalist.append(dataFrame.etData[CN])
    
    for i in range(dslen):
        if(i<len(datalist[0])):
            df[i,0] = [dataFrame.etData[classes[0]][i].filename+".jpg"]
            filenames.append(dataFrame.classes[classes[0]]+"_"+dataFrame.etData[classes[0]][i].filename+".jpg")
            df[i,1] = dataFrame.etData[classes[0]][i].gtbb
            df[i,2] = classes[0]
            targets[i] =  dataFrame.etData[classes[0]][i].gtbb
            classlabels[i] = classes[0]
            imdims[i] = dataFrame.etData[classes[0]][i].dimensions[:2] #some classes also have channel-dim
            for j in range(dataFrame.NUM_TRACKERS):
                sliceShape = dataFrame.eyeData[classes[0]][i][j][0].shape
                df[i,3+j] = dataFrame.eyeData[classes[0]][i][j][0] #list items
                if(sliceShape != (2,0)): #for all non-empty
                    eyes[i,j,:sliceShape[0],:] = torch.from_numpy(dataFrame.eyeData[classes[0]][i][j][0][-32:].astype(np.int32)) #some entries are in uint16, which torch does not support
                else: 
                    eyes[i,j,:,:] = 0.0
                    #print("error-filled measurement [entryNo,participantNo]: ",i,",",j)
                    #print(eyes[i,j])
                
                    
                #old: padding too early #eyes[i,j] = zero_pad(dataFrame.eyeData[classes[0]][i][j][0],num_points,-999) #list items
                
                
        else: 
            nextIdx = i-len(datalist[0])
            df[i,0] = [dataFrame.etData[classes[1]][nextIdx].filename+".jpg"]
            filenames.append(dataFrame.classes[classes[1]]+"_"+dataFrame.etData[classes[1]][nextIdx].filename+".jpg")
            df[i,1] = dataFrame.etData[classes[1]][nextIdx].gtbb
            targets[i] =  dataFrame.etData[classes[1]][nextIdx].gtbb
            df[i,2] = classes[1]
            classlabels[i] = classes[1]
            imdims[i] = dataFrame.etData[classes[1]][nextIdx].dimensions
            for j in range(dataFrame.NUM_TRACKERS):
                sliceShape = dataFrame.eyeData[classes[1]][nextIdx][j][0].shape
                df[i,3+j] = dataFrame.eyeData[classes[1]][nextIdx][j][0]
                if(sliceShape != (2,0)): #for empty
                    eyes[i,j,:sliceShape[0],:] = torch.from_numpy(dataFrame.eyeData[classes[1]][nextIdx][j][0][-32:]) #last 32 points
                else:
                    eyes[i,j,:,:] = 0.0
                
                #old: padding too early #eyes[i,j] = zero_pad(dataFrame.eyeData[classes[1]][nextIdx][j][0],num_points,-999) #list items
                
            
        
        
    print("Total length is: ",dslen)
    
    targetsTensor = torch.from_numpy(targets)
    classlabelsTensor = torch.from_numpy(classlabels)
    imdimsTensor = torch.from_numpy(imdims)
    
    return df,filenames,eyes,targetsTensor,classlabelsTensor,imdimsTensor,dslen

class PASCALdataset(Dataset):
    def __init__(self,pascalobj,root_dir,classes,sTransform=None,imTransform = None,coordTransform = None):
        self.ABDset,self.filenames,self.eyeData,self.targets,self.classlabels,self.imdims,self.length = generate_DS(pascalobj,classes)
        self.root_dir = root_dir
        self.sTransform = sTransform
        self.imTransform = imTransform
        self.coordTransform = coordTransform
        self.deep_representation = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ViT_model = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=0, global_pool='').to(device)
        
    def __len__(self): 
        return len(self.ABDset)
    
    
    def __getitem__(self,idx): #for images
        if torch.is_tensor(idx):
            idx = idx.tolist() #for indexing
        
        SINGLE_PARTICIPANT = True
        
        
        filename = self.filenames[idx]
        image = Image.open(self.root_dir + filename)
        
       
        if(SINGLE_PARTICIPANT==True):
            signals = self.eyeData[idx][0] #participant zero only
        
        else: 
            signals = self.eyeData[idx,:]
        

        
        targets = self.targets[idx,:]
        
        sample = {"signal": signals,"target": targets,"file": filename, "index": idx,"size":self.imdims[idx],"class":self.classlabels[idx],"mask":None,"image":image,"orgimage":image,"scaled_signal":None,"scaled_target":None,"target_centered":None}
        
        if self.sTransform: 
            sample = self.sTransform(sample)
        if self.imTransform:
            sample["orgimage"] = self.imTransform(sample["image"]) #only shoot through image 
        
        if self.coordTransform:
            sample = self.coordTransform(sample)
        
        if sample["orgimage"].dim()==3:
            sample["image"] = sample["orgimage"].unsqueeze(0)
            
        #with torch.no_grad():    
            #sample["image"] = self.ViT_model.forward_features(sample["image"])
            #should interpolate here instead of later, might aswell save lines of code later on
            
        return sample
    
def get_split(root_dir,classesOC):
    """
    Function to get professors splits
    
        Args: 
        root_dir, ie. path to main-project-folder
        classesOC: List of classes of interest
        
        Returns:
        list of filenames for training
        list of filenames for testing
    """
        
    split_dir = root_dir +"/Data/POETdataset/split/"
    classlist = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]

    training_data = []
    training_len_list = []
    for classes in classesOC: #loop saves in list of lists format
        train_filename = split_dir + classlist[classes]+"_Rbbfix.txt"
        with open(train_filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        for i in range(len(lines)): 
            lines[i] = classlist[classes]+"_"+lines[i]+".jpg"
            
        training_data.append(lines)
        training_len_list.append(len(lines)) #append length of dataset to a list, for balanced overfit-sampling
        
    test_data = []
    for classes in classesOC: 
        train_filename = split_dir + classlist[classes]+"_Rfix.txt"
        with open(train_filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        for i in range(len(lines)): 
            lines[i] = classlist[classes]+"_"+lines[i]+".jpg"
            
        test_data.append(lines)
    
    #unpack lists of lists to single list for train and for test
    train = []
    test = []
    for i in range(len(training_data)):
        train = [*train,*training_data[i]]
        test = [*test,*test_data[i]]        
         
    return train,test,training_len_list

def load_split(className,root_dir):
    #loads earlier saved splits from disk. 
    #Arg: className (e.g. "airplane")
    
    tmp_root = root_dir+"/datasets/"
    print("................. Loaded ",className," datasets from disk .................")
    train = torch.load(tmp_root+className+"Train.pt") #load earlier saved split
    test = torch.load(tmp_root+className+"Test.pt")
    train_len_li = torch.load(tmp_root+className+"train_length_list.pt") #only used for balanced sampling when overfitting multiple class dset
    return train,test,train_len_li


def get_balanced_permutation(nsamples,NUM_IN_OVERFIT):
    sample_perc = [x/sum(nsamples) for x in nsamples] #get percentage-distribution 
    if(len(nsamples)>1):
        classwise_nsamples = [math.ceil(x*NUM_IN_OVERFIT) for x in sample_perc] #get number of samples per class
        NUM_IN_OVERFIT = sum(classwise_nsamples) #overwrite NUM_IN_OVERFIT, as you use math.ceil
    else: 
        classwise_nsamples = [NUM_IN_OVERFIT]
        
    h = [torch.Generator() for i in range(len(classesOC))]
    for i in range(len(classesOC)):
        h[i].manual_seed(i)
        
    ofIDX = torch.zeros(0).to(torch.int32)
    valIDX = torch.zeros(0).to(torch.int32)
    offset = 0
    for num,instance in enumerate(nsamples):
        idx = torch.randperm(int(nsamples[num]),generator=h[num])
        t_idx = idx + offset #add per-class offset for classwise indexing
        idx = t_idx[:classwise_nsamples[num]]
        vidx = t_idx[classwise_nsamples[num]:]
        ofIDX = torch.cat((ofIDX,idx),0)
        valIDX = torch.cat((valIDX,vidx),0)
        offset += instance
        
    return ofIDX,valIDX,NUM_IN_OVERFIT

def rescale_coordsO(eyeCoords,imdims,bbox):
    """
    Args: 
        eyeCoords: Tensor
        imdims: [1,2]-tensor with [height,width] of image
        imW: int value with width of image 
        imH: int value of height of image 
    Returns: 
        Rescaled eyeCoords (inplace) from [0,1].
    """
    outCoords = torch.zeros_like(eyeCoords)
    outCoords[:,0] = eyeCoords[:,0]/imdims[1]
    outCoords[:,1] = eyeCoords[:,1]/imdims[0]
    
    tbox = torch.zeros_like(bbox)
    #transform bounding-box to [0,1]
    tbox[0] = bbox[0]/imdims[1]
    tbox[1] = bbox[1]/imdims[0]
    tbox[2] = bbox[2]/imdims[1]
    tbox[3] = bbox[3]/imdims[0]
    
    
    return outCoords,tbox
   

class tensorPad(object):
    """
    Object-function which pads a batch of different sized tensors to all have [bsz,32,2] with value -999 for padding

    Parameters
    ----------
    object : lambda, only used for composing transform.
    sample : with a dict corresponding to dataset __getitem__

    Returns
    -------
    padded_batch : tensor with dimensions [batch_size,32,2]

    """
    def __call__(self,sample):
        x = sample["signal"]
        xtmp = torch.zeros(32,2)
        numel = x.shape[0]
        xtmp[:numel,:] = x[:,:]
        
        maskTensor = (xtmp[:,0]==0)    
        sample["signal"] = xtmp
        sample["mask"] = maskTensor
        return sample
     
class rescale_coords(object):
    """
    Args: 
        eyeCoords: Tensor
        imdims: [1,2]-tensor with [height,width] of image
        imW: int value with width of image 
        imH: int value of height of image 
    Returns: 
        Rescaled eyeCoords (inplace) from [0,1].
    """
    
    def __call__(self,sample):
        eyeCoords,imdims,bbox = sample["signal"],sample["size"],sample["target"]
        mask = sample["mask"]
        outCoords = torch.ones(32,2)
        
        #old ineccesary #eyeCoords[inMask] = float('nan') #set nan for next calculation -> division gives nan. Actually ineccesary
        #handle masking
        #inMask = (eyeCoords==0.0) #mask-value
        #calculate scaling
        outCoords[:,0] = eyeCoords[:,0]/imdims[1]
        outCoords[:,1] = eyeCoords[:,1]/imdims[0]
        
        #reapply mask
        outCoords[mask] = 0.0 #reapply mask
        
        tbox = torch.zeros_like(bbox)
        #transform bounding-box to [0,1]
        tbox[0] = bbox[0]/imdims[1]
        tbox[1] = bbox[1]/imdims[0]
        tbox[2] = bbox[2]/imdims[1]
        tbox[3] = bbox[3]/imdims[0]
        
        sample["scaled_signal"] = outCoords
        sample["scaled_target"] = tbox
        
        return sample
    
        #return {"signal": outCoords,"size":imdims,"target":tbox}

class get_center(object):
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
    def __call__(self,sample):
        targets = sample["scaled_target"]
        assert targets!=None, print("FOUND NO SCALED TARGET")
        xywh = torch.zeros(4,dtype=torch.float32)
        #w = torch.zeros(targets.shape[0],dtype=torch.int32)
        #h = torch.zeros(targets.shape[0],dtype=torch.int32)
        
        xywh[0]  = torch.div(targets[2]+targets[0],2)
        xywh[1] = torch.div(targets[-1]+targets[1],2)
        xywh[2] = targets[2]-targets[0]
        xywh[3] = targets[-1]-targets[1]
        sample["target_centered"] = xywh
        return sample


def center_to_box(targets):
    """
    Scales center-prediction cx,cy,w,h back to x0,y0,y1,y2

    Parameters
    ----------
    targets : batched torch tensor [BS,4]
        Format center x, center y, width and height of box

    Returns
    -------
    box_t : batched torch tensor
        Format lower left corner, upper right corner, [x0,y0,x1,y1]
        
    """
    box_t = torch.zeros(targets.shape[0],4,dtype=torch.float32)
    box_t[:,0] = targets[:,0]-torch.div(targets[:,2],2) #x0
    box_t[:,1] = targets[:,1]-torch.div(targets[:,3],2) #y0
    box_t[:,2] = targets[:,0]+torch.div(targets[:,2],2) #x1
    box_t[:,3] = targets[:,1]+torch.div(targets[:,3],2) #y2
    return box_t

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]
GENERATE_DATASET = True
if((len(classesOC)>1) and (len(classesOC)<10)):
    classString = "multiple_classes"
elif(len(classesOC)==10):
    classString = "all_classes"
else: 
    classString = classes[classesOC[0]]
    
timm_root = os.path.dirname(__file__)
BATCH_SZ = 8
NLAYERS = 1
NHEADS = 1
EPOCHS = 3
NUM_IN_OVERFIT = 200
DROPOUT = 0.0
LR_FACTOR = 1
EPOCH_STEPS = NUM_IN_OVERFIT//BATCH_SZ
if EPOCH_STEPS == 0: 
    EPOCH_STEPS = 1
#NUM_WARMUP = int(EPOCHS*(1/3))*EPOCH_STEPS #constant 30% warmup-rate
NUM_WARMUP = 1
BETA = 1
OVERFIT=True
EVAL = 0 #flag which is set for model evaluation, ie. final model. Set 1 if it is final model.

   

        
        


    
if(GENERATE_DATASET == True or GENERATE_DATASET==None):
    dataFrame = pascalET()
    #root_dir = os.path.dirname(__file__) + "/../eyeFormer/Data/POETdataset/PascalImages/"
    root_dir = os.path.join(os.path.expanduser('~'),"/home/max/Documents/s194119/New_bachelor/Data/POETdataset/PascalImages/")
    split_root_dir = os.path.join(os.path.expanduser('~'),"/home/max/Documents/s194119/New_bachelor")
    
    SignalTrans = torchvision.transforms.Compose(
        [torchvision.transforms.Lambda(tensorPad())])
        #,torchvision.transforms.Lambda(rescale_coords())]) #transforms.Lambda(tensor_pad() also a possibility, but is probably unecc.
    
    ImTrans = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224),torchvision.transforms.InterpolationMode.BILINEAR),
                                              torchvision.transforms.ToTensor()]) #should be normalized and converted to 0,1. ToTensor always does this for PIL images
    
    #EXPERIMENTARY
    CoordTrans = torchvision.transforms.Compose([torchvision.transforms.Lambda(rescale_coords()),torchvision.transforms.Lambda(vm.get_center())])
    
    fullDataset = PASCALdataset(dataFrame, root_dir, classesOC,sTransform=SignalTrans,imTransform = ImTrans,coordTransform = CoordTrans) #init dataset as torch.Dataset.
    
    trainLi,testLi,nsamples = get_split(split_root_dir,classesOC) #get Dimitrios' list of train/test-split.
    trainIDX = [fullDataset.filenames.index(i) for i in trainLi] #get indices of corresponding entries in data-structure
    testIDX = [fullDataset.filenames.index(i) for i in testLi]
    #subsample based on IDX
    train = torch.utils.data.Subset(fullDataset,trainIDX)
    test = torch.utils.data.Subset(fullDataset,testIDX)
    #save generated splits to scratch
    torch.save(train,timm_root+"/datasets/"+classString+"Train.pt")
    torch.save(test,timm_root+"/datasets/"+classString+"Test.pt")
    torch.save(nsamples,timm_root+"/datasets/"+classString+"train_length_list.pt") #only used for balanced sampling when overfitting
    
    
    print("................. Wrote datasets for ",classString,"to disk ....................")
    

if(GENERATE_DATASET == False):
    train,test = load_split(classString,timm_root)  
    print("Succesfully read data from binary")
 
#FOR DEBUGGING
#train = torch.utils.data.Subset(train,[0,1,2])


#make dataloaders of chosen split
trainloader = DataLoader(train,batch_size=BATCH_SZ,shuffle=True,num_workers=0)
#FOR DEBUGGING
#for i, data in enumerate(trainloader):
#    if i==1:
#        DEBUGSAMPLE = data

testloader = DataLoader(test,batch_size=BATCH_SZ,shuffle=True,num_workers=0)

if NUM_IN_OVERFIT==None and OVERFIT==True: #only if no cmd-argv provided
    NUM_IN_OVERFIT=16
    print("...No argument for length of DSET provided. Used N=16...")
    
g = torch.Generator()
g.manual_seed(8)

if(OVERFIT): #CREATES TRAIN AND VALIDATION-SPLIT 
    
    #new mode for getting representative subsample: 
    ofIDX,valIDX,NUM_IN_OVERFIT = get_balanced_permutation(nsamples,NUM_IN_OVERFIT)
    
    #IDX = torch.randperm(len(train),generator=g[0])#[:NUM_IN_OVERFIT].unsqueeze(1) #random permutation, followed by sampling and unsqueezing
    #ofIDX = IDX[:NUM_IN_OVERFIT].unsqueeze(1)
    #valIDX = IDX[NUM_IN_OVERFIT:NUM_IN_OVERFIT+17].unsqueeze(1) #17 because the smallest train-set has length 33=max(L)+17.
    overfitSet = torch.utils.data.Subset(train,ofIDX)
    valSet = torch.utils.data.Subset(train,valIDX)
    trainloader = DataLoader(overfitSet,batch_size=BATCH_SZ,shuffle=True,num_workers=0,generator=g)
    print("Overwrote trainloader with overfit-set of length {}".format(ofIDX.shape[0]))
    valloader = DataLoader(valSet,batch_size=BATCH_SZ,shuffle=True,num_workers=0,generator=g)
    print("Valloader of constant length {}".format(valIDX.shape[0]))


#debug sizes of testloader 
for i, data in enumerate(testloader):
    BS = data["size"].shape[0]
    for j in range(BS):
        w = data["size"][j][1].item()
        h = data["size"][j][0].item()
        if (int(w),int(h)) == (0,0):
            print("DETECTED EMPTY SHAPE FOR SAMPLE {} in BATCH {}, with FILENAME {}".format(j,i,data["file"][j]))
    #if i==1: 
     #   break