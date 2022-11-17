#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:03:36 2022

@author: max
"""
from load_POET import pascalET
import os 
from torch.utils.data import Dataset, DataLoader
import torch 
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import transforms, ops
import math
from sklearn.model_selection import KFold
from tqdm import tqdm


class AirplanesBoatsDataset(Dataset):
    def __init__(self,pascalobj,root_dir,classes,transform=None):
        """old init: 
            def __init__(self,pascalobj,root_dir,classes):#,eyeData,filenames,targets,classlabels,root_dir,classes):
                self.ABDset = generate_DS(pascalobj,classes)
                self.eyeData = eyeData
                self.filenames = filenames
                self.root_dir = root_dir
                self.targets = targets #bbox
                self.classlabels = classlabels
        """

        self.ABDset,self.filenames,self.eyeData,self.targets,self.classlabels,self.imdims,self.length = generate_DS(pascalobj,classes)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self): 
        return len(self.ABDset)
    
    
    def __getitem__(self,idx): #for images
        if torch.is_tensor(idx):
            idx = idx.tolist() #for indexing
        
        SINGLE_PARTICIPANT = True
        
        #PROBABLY THE PROBLEM IS YOU ARE SETTING THE VARS FROM NUMPY "OBJECT"-TYPE
        #filename = self.ABDset[idx,0]
        filename = self.filenames[idx]
        #target = self.ABDset[idx,1].astype(np.int16) #earlier issue: cant convert type np.uint8
        #objClass = self.ABDset[idx,2]
        #signals = self.ABDset[idx,3:]
       
        if(SINGLE_PARTICIPANT==True):
            signals = self.eyeData[idx][0] #participant zero only
        
        else: 
            signals = self.eyeData[idx,:]
        
        targets = self.targets[idx,:]
        
        sample = {"signal": signals,"target": targets,"file": filename, "index": idx,"size":self.imdims[idx],"class":self.classlabels[idx],"mask":None}
        
        if self.transform: 
            sample = self.transform(sample)
        
        return sample
        
                

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
        
        sample["signal"] = outCoords
        sample["target"] = tbox
        
        return sample
        #return {"signal": outCoords,"size":imdims,"target":tbox}
    
    
    


"""FOR IMAGES
    def __getitem__(self,idx): #for images
        if torch.is_tensor(idx):
            idx = idx.tolist() #for indexing
        
        classdict = {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}
        className = classdict[self.ABDset[idx,2]]+"_"
        img_name = self.root_dir+className+ self.ABDset[idx,0]
        print(img_name)
"""
    
    
    
def generate_DS(dataFrame,classes=[x for x in range(10)]):
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
            imdims[i] = dataFrame.etData[classes[0]][i].dimensions
            for j in range(dataFrame.NUM_TRACKERS):
                sliceShape = dataFrame.eyeData[classes[0]][i][j][0].shape
                df[i,3+j] = dataFrame.eyeData[classes[0]][i][j][0] #list items
                if(sliceShape != (2,0)): #for empty
                    eyes[i,j,:sliceShape[0],:] = torch.from_numpy(dataFrame.eyeData[classes[0]][i][j][0][-32:])
                else: 
                    eyes[i,j,:,:] = 0.0
                    print("error-filled measurement [entryNo,participantNo]: ",i,",",j)
                    print(eyes[i,j])
                
                    
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
   


def zero_pad(inArr: np.array,padto: int,padding: int):
    """
    Args:
        inputs: 
            inArr: array to pad
            padto: integer value of the array-size used. Uses last padto elements of eye-signal
            padding: integer value of padding value, defaults to zero
            
        output: 
            torch.tensor of dtype torch.float32 with size [padto,2]
        
    """
    if(inArr.shape[0]>padto):
        return torch.from_numpy(inArr[:padto]).type(torch.float32)
        
    else: 
        outTensor = torch.ones((padto,2)).type(torch.float32)*padding #[32,2]
        if(inArr.shape[1]==0): #Case: no valid entries in eyeData
            return outTensor
        
        numEntries = inArr.shape[0]
        outTensor[:numEntries,:] = torch.from_numpy(inArr[-numEntries:,:]) #get all
        return outTensor
    

#-------------------------------------SCRIPT PARAMETERS---------------------------------------#
torch.manual_seed(9)
CHECK_BALANCE = False
GENERATE_DATASET = False
OVERFIT = True
NUM_IN_OVERFIT = 16
classString = "airplanes"
SAVEFIGS = True
BATCH_SZ = 1
<<<<<<< HEAD
EPOCHS = 2000
VAL_PERC = 0.3 #length of validation set 
=======
EPOCHS = 250
VAL_PERC = 0.25 #length of validation set 
>>>>>>> b01656d64f9c5616efef19dab93f1b57da8d5479
#-------------------------------------SCRIPT PARAMETERS---------------------------------------#

if(GENERATE_DATASET == True):
    dataFrame = pascalET()
    root_dir = os.path.dirname(__file__) + "/Data/POETdataset/PascalImages/"
    
    """Old implementation, corresponding to commented part in dataset-init:
    #DF,FILES,EYES,TARGETS,CLASSLABELS, IMDIMS = generate_DS(dataFrame,[0,9]) #init DSET, extract interesting stuff as tensors. Aim for making DS obsolete and delete
    #airplanesBoats = AirplanesBoatsDataset(dataFrame, EYES, FILES, TARGETS,CLASSLABELS, root_dir, [0,9]) #init dataset as torch.Dataset.
    """
    
    classesOC = [0]
    composed = transforms.Compose([transforms.Lambda(tensorPad()),
                                   transforms.Lambda(rescale_coords())]) #transforms.Lambda(tensor_pad() also a possibility, but is probably unecc.
    
    airplanesBoats = AirplanesBoatsDataset(dataFrame, root_dir, classesOC,transform=composed) #init dataset as torch.Dataset.
    
    
    
    #"""
    #check that transforms work 
    
    #define split ratio
    test_size = int(0.2*airplanesBoats.length)
    train_size = airplanesBoats.length-test_size
    
    #set seed 
    
    #get split 
    train,test = torch.utils.data.random_split(airplanesBoats,[train_size,test_size])
    ###load split from data
    
    split_root_dir = os.path.dirname(__file__)

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
    for classes in classesOC: #loop saves in list of lists format
        train_filename = split_dir + classlist[classes]+"_Rbbfix.txt"
        with open(train_filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        for i in range(len(lines)): 
            lines[i] = classlist[classes]+"_"+lines[i]+".jpg"
            
        training_data.append(lines)
        
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
         
    return train,test

root_dir = os.path.dirname(__file__) + "/Data/POETdataset/"
if(GENERATE_DATASET == True):
    trainLi,testLi = get_split(split_root_dir,classesOC) #get Dimitrios' list of train/test-split.
    trainIDX = [airplanesBoats.filenames.index(i) for i in trainLi] #get indices of corresponding entries in data-structure
    testIDX = [airplanesBoats.filenames.index(i) for i in testLi]
    #subsample based on IDX
    train = torch.utils.data.Subset(airplanesBoats,trainIDX)
    test = torch.utils.data.Subset(airplanesBoats,testIDX)
    root_dir = os.path.dirname(__file__) + "/Data/POETdataset/"
    torch.save(train,root_dir+classString+"Train.pt")
    torch.save(test,root_dir+classString+"Test.pt")
    print("................. Wrote datasets for ",classString,"to disk ....................")

def load_split(className):
    print("................. Loaded ",className," datasets from disk .................")
    train = torch.load(root_dir+className+"Train.pt")
    test = torch.load(root_dir+className+"Test.pt")
    return train,test

if(GENERATE_DATASET == False):
    train,test = load_split(classString)
    


#make dataloaders of chosen split

trainloader = DataLoader(train,batch_size=BATCH_SZ,shuffle=True,num_workers=0)
testloader = DataLoader(test,batch_size=BATCH_SZ,shuffle=True,num_workers=0)
#for model overfitting
ofIDX = torch.randperm(len(train))[:NUM_IN_OVERFIT].unsqueeze(1) #random permutation, followed by sampling and unsqueezing
overfitSet = torch.utils.data.Subset(train,ofIDX)
oTrainLoader = DataLoader(overfitSet,batch_size=BATCH_SZ,shuffle=True,num_workers=0)

""" Unused section for making a train/val-split on overfit-set
train_idx = torch.randint(0,len(train),(NUM_IN_OVERFIT,1))
val_idx = torch.arange(0,len(train)).reshape(len(train),1)
mask = [i not in train_idx for i in val_idx] #remove entries in train_idx from val_idx
val_idx = val_idx[mask] #remask

overfitSet = torch.utils.data.Subset(train,train_idx)
overfitValSet = torch.utils.data.Subset(train,val_idx)
oTrainLoader = DataLoader(overfitSet,batch_size=BATCH_SZ,shuffle=True,num_workers=0)
oValLoader = DataLoader(overfitSet,batch_size=BATCH_SZ,shuffle=True,num_workers=0)

"""


#check class-balance (only necessary for finding appropriate seed ONCE): 
CHECK = False
if CHECK:
    CHECK_BALANCE()   

def CHECK_BALANCE():
    if(CHECK_BALANCE==True):
        countDict = {"trainplane":0,"trainsofa":0,"testplane":0,"testsofa":0}
        for i_batch, sample_batched in enumerate(trainloader):
            for j in range(sample_batched["class"].shape[0]):
                if sample_batched["class"][j]==0:
                    countDict["trainplane"] += 1 
                if sample_batched["class"][j]==9:
                    countDict["trainsofa"] += 1
                
        for i_batch, sample_batched in enumerate(testloader):
            for j in range(sample_batched["class"].shape[0]):
                if sample_batched["class"][j]==0:
                    countDict["testplane"] += 1 
                if sample_batched["class"][j]==9:
                    countDict["testsofa"] += 1
    else:
        return
        
    
    print("Train-set: (airplanes, sofas) (",countDict["trainplane"]/train_size,"),(",countDict["trainsofa"]/train_size,")")
    print("Test-set: (airplanes, sofas) (",countDict["testplane"]/test_size,"),(",countDict["testsofa"]/test_size,")")
    print("Seems quite balanced :-)")


"""
Model definitions: 
and build

"""    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class eyeFormer_baseline(nn.Module):
        def __init__(self,input_dim=2,hidden_dim=2048,output_dim=4,dropout=0.1):
            self.d_model = input_dim
            super().__init__() #get class instance
            #self.embedding = nn.Embedding(32,self.d_model) #33 because cls needs embedding
            self.pos_encoder = PositionalEncoding(self.d_model,dropout)
            self.encoder = TransformerEncoder(1,input_dim=self.d_model,seq_len=32,num_heads=1,dim_feedforward=hidden_dim) #False due to positional encoding made on batch in middle
            #make encoder - 3 pieces
            self.cls_token = nn.Parameter(torch.zeros(1,self.d_model),requires_grad=True)
            #self.transformer_encoder = nn.TransformerEncoder(encoderLayers,num_layers = 1)
            
            
            #self.decoder = nn.Linear(64,4,bias=True) 
            self.clsdecoder = nn.Linear(self.d_model,4,bias=True)
            self.DEBUG = False
        
        def switch_debug(self):
            self.DEBUG = not self.DEBUG
            if(self.DEBUG==True):
                string = "on"
            else: string = "off"
            print("Debugging mode turned "+string)
            
            
        def forward(self,x,src_padding_mask=None):    
            if x.dim()==1: #fix for batch-size 1 
                x = x.unsqueeze(0)
            
            bs = x.shape[0]

            if src_padding_mask==None: 
                src_padding_mask = torch.zeros(bs,x.shape[1]).to(dtype=torch.bool)
            #print("key-mask\n",src_padding_mask)
            clsmask = torch.zeros(bs,1).to(dtype=torch.bool)
            mask = torch.cat((clsmask,src_padding_mask[:,:].reshape(bs,32)),1) #unmask cls-token
           
            x = x* math.sqrt(self.d_model) #as this in torch tutorial but dont know why
            x = torch.cat((self.cls_token.expand(x.shape[0],1,2),x),1) #concat along sequence-dimension. Copy bs times
            if self.DEBUG==True:
                print("2: scaled and cat with CLS:\n",x,x.shape)
            x = self.pos_encoder(x)
            if self.DEBUG==True:
                print("3: positionally encoded: \n",x,x.shape)
            
            #print("Src_padding mask is: ",src_padding_mask)
            #print("pos encoding shape: ",x.shape)
            output = self.encoder(x,mask)
            if self.DEBUG==True:
                print("4: Transformer encoder output:\n",output)
            #print("encoder output:\n",output)
            #print("Encoder output shape:\n",output.shape)
            #print("Same as input :-)")
           
            output = self.clsdecoder(output[:,0,:]) #batch-first is true. Picks encoded cls-token-vals for the batch.
            if self.DEBUG==True:
                print("5: linear layer based on CLS-token output: \n",output)
            
            return output
            
           
            


class EncoderBlock(nn.Module):
        
    """
    Inputs:
        input_dim - Dimensionality of the input
        num_heads - Number of heads to use in the attention block
        dim_feedforward - Dimensionality of the hidden layer in the MLP
        dropout - Dropout probability to use in the dropout layers
    """         
    def __init__(self,input_dim,seq_len,num_heads,dim_feedforward,dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(input_dim,num_heads,dropout,batch_first=True)
        
        #two-layer FFN
        self.linear_net = nn.Sequential(nn.Linear(input_dim,dim_feedforward),
                                        nn.Dropout(dropout),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(dim_feedforward,input_dim))
        #layer normalization: 
        self.norm1 = nn.LayerNorm((seq_len+1,input_dim))
        self.norm2 = nn.LayerNorm((seq_len+1,input_dim))
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):
       # Attention part
       attn_out, _ = self.self_attn(x,x,x,key_padding_mask=mask)
       x = x + self.dropout(attn_out)
       x = self.norm1(x)

       # MLP part
       linear_out = self.linear_net(x)
       x = x + self.dropout(linear_out)
       x = self.norm2(x)

       return x
     

class TransformerEncoder(nn.Module):
    def __init__(self,num_layers,**block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])
        
    def forward(self,x,mask=None):
        for l in self.layers: 
            x = l(x,mask)
        return x 
    
    
     
class PositionalEncoding(nn.Module):
    ###Probably change max_len of pos-encoding
    def __init__(self,d_model,dropout = 0.1,max_len = 33):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe = torch.zeros(1,max_len,d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term) #first dimension is batch dimension
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Returns nn.Dropout(x+pe(x)). Parse batch-first
        
        
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            
        Returns: 
            PosEnc(x): Tensor, shape [batchsize,seq_len,embedding_dim]
        """
        x = x + self.pe[:,:x.size(1),:] #[bs x seq_len x embedding dim]
        return self.dropout(x)

def scaleBackCoords(preds,labels,imdims):
    """
    Function for scaling from percentage back to original box-coords. Necessary for correct IOU-calculation. Should work for batches 
    Input: 
        percentage-wise preds and labels in format [x0,y0,x1,y1]
        imdims: [1,2]-tensor holding image-size in format [height,width]
    returns: 
        preds, labels in pixel-values 
        
    """
    BSZ = preds.shape[0]
    tmp_preds = torch.zeros_like(preds)
    tmp_labels = torch.zeros_like(labels)
    
    assert BSZ == labels.shape[0], "zeroth dimension of preds and labels are not the same"
    for i in range(BSZ):
        tmp_preds[i,0::2] = preds[i,0::2]*imdims[i,-1] #x 
        tmp_preds[i,1::2] = preds[i,1::2]*imdims[i,0] #y
        tmp_labels[i,0::2] = labels[i,0::2]*imdims[i,-1]  
        tmp_labels[i,1::2] = labels[i,1::2]*imdims[i,0] 
    return tmp_preds,tmp_labels 

def boxIOU(preds,labels):
    """
    Own-implemented batch-solid IOU-calculator. 
    Returns: tensor, [BS]
    """
    BSZ = preds.shape[0]
    assert BSZ == labels.shape[0], "zeroth dimension of preds and labels are not the same"
    IOU = torch.zeros(BSZ)
    for i in range(BSZ):
        A_target = (labels[i,2]-labels[i,0])*(labels[i,3]-labels[i,1]) #(x2-x1)*(y2-y1)
        A_pred = (preds[i,2]-preds[i,0])*(preds[i,-1]-preds[i,1]) #(x2-x1)*(y2-y1)
        U_width = torch.min(labels[i,2],preds[i,2]) - torch.max(labels[i,0],preds[i,0]) #width is min(lx2,px2)-(max(lx0,px0))
        U_height = torch.min(labels[i,3],preds[i,3]) - torch.max(labels[i,1],preds[i,1])  
        A_U = U_width * U_height
        IOU[i] = A_U / (A_target+A_pred-A_U)
    return IOU      
    
def pascalACC(preds,labels): #TODO: does not work for batched input. Fix
    """
    Function for calculating the accuracy between a batch of predictions and corresponding batch of targets. 
    Returns: number of correct predictions in batch, number of false predictions in batch and a list of IOU-scores for the batch
    """
    
    BSZ = preds.shape[0]
    assert BSZ == labels.shape[0],"Batch-size dimensions between target and tensor not in corresondance!"
    
    no_corr = 0 
    no_false = 0
    IOU_li = []
    
    if preds.dim()==1:
        preds = preds.unsqueeze(0)
    if labels.dim()==1:
        labels = labels.unsqueeze(0)
        
    #no loop approach: 
        #may be more effective for small batches. But only the diagonal of IOU_tmp is of interest for you - thus many wasted calculations
    #IOU_tmp = ops.box_iou(preds,labels) #calculates pairwise 
    #print(torch.diagonal(IOU_tmp))
    
    for i in range(BSZ): #get pascal-criterium accuraccy
        pred_tmp = preds[i,:].unsqueeze(0)
        label_tmp = labels[i,:].unsqueeze(0)
        IOU = ops.box_iou(pred_tmp,label_tmp)
        IOU_li.append(IOU.item())
        if(IOU>0.5):
            no_corr += 1
        else:
            no_false += 1

    return no_corr,no_false,IOU_li



###-----------------------------------MODEL TRAINING----------------------------------
model = eyeFormer_baseline()
activation = {}
def getActivation(name):
    def hook(model,input,output):
        activation[name] = output.detach()
    return hook
#register forward hooks
#h1 = model.encoder.layers[0].self_attn.register_forward_hook(getActivation('MHA'))
h2 = model.clsdecoder.register_forward_hook(getActivation('CLS_lin'))
h3 = model.encoder.layers[0].linear_net[1].register_forward_hook(getActivation('before_relu'))
h4 = model.encoder.layers[0].linear_net[2].register_forward_hook(getActivation('after_relu'))
h5 = model.encoder.layers[0].linear_net[2].register_forward_hook(getActivation('after_lin_norm'))
h6 = model.encoder.layers[0].norm2.register_forward_hook(getActivation('MHA_norm'))

def checkDeadRelu(t,show=False):
    """
    Parameters
    ----------
    t : pyTorch tensor
        shape: [bs,CLS(1)+seq_len,hidden_dim]

    Returns
    -------
    tensor : [number of negative elements, total number of elements (all batches)]
    """
    flat = t.reshape(-1)
    num_no_active = (flat<=0).sum(0)
    if(show==True):
        print("Number of not activated neurons: {}/{}".format(num_no_active,t.numel()))
    return torch.tensor([num_no_active,t.numel()])

def getIOU(preds,target,sensitivity=0.5): #todo: fix for bigger batches
    """Evaluates IOU of predictions and target, returns no of correct and falses, and a list of IOU vals"""
    correct_count = 0 
    false_count = 0
    IOU_li = []
    for i in range(target.shape[0]): #get pascal-criterium 
        IOU = ops.box_iou(preds[i].unsqueeze(0),target[i].unsqueeze(0)) #outputs: Nx4, data["target"]: Mx4
        IOU_li.append(IOU)
        #print("IOU IS:",IOU)
        if(IOU>sensitivity):
            correct_count += 1
        else:
            false_count += 1
    return correct_count,false_count,IOU_li

#model.switch_debug()

#=================DEFINE OPTIMIZER ================#
class NoamOpt:
    #"Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
    def get_std_opt(model):
        return NoamOpt(model.d_model, 2, 4000,torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
###

#optimizer = torch.optim.Adam(model.parameters(),lr=0.0001) #[2e-2,2e-4,2e-5,3e-5,5e-5]
model_opt = NoamOpt(model.d_model,0.8,120,torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

loss_fn = nn.SmoothL1Loss(beta=1) #default: mean and beta=1.0

encoder_list,linear_list,lin1_list,lin2_list = [], [],[],[]
dead_neurons_lin1 = []
dead_neurons_lin2 = []

def train_one_epoch(model,loss,trainloader,oTrainLoader,overfit=False,negative_print=False) -> float: 
    running_loss = 0.
    #src_mask = generate_square_subsequent_mask(32).to(device)
    correct_count = 0
    false_count = 0
    counter = 0
    
    
    if(overfit==True):
        trainloader = oTrainLoader
    else: 
        trainloader = trainloader #a bit clumsy, may be refactored
        
    
    for i, data in enumerate(trainloader):
        counter += 1
        model_opt.optimizer.zero_grad() #reset grads
        target = data["target"]
        mask = data["mask"]
        imsz = data["size"]
        #print("Mask:\n",data["mask"])
        #print("Input: \n",data["signal"])
        #print("Goal is: \n",data["target"])
        outputs = model(data["signal"],mask)
        
        #register activations 
        #encoder_list.append(activation['MHA'])
        linear_list.append(activation["CLS_lin"])
        lin1_list.append(activation["before_relu"])
        lin2_list.append(activation["after_relu"])
        #dead_neurons_lin1.append(checkDeadRelu(activation["before_relu"]))
        #dead_neurons_lin2.append(checkDeadRelu(activation["after_relu"]))
        if(negative_print==True):
            print("Number of negative entries [dead-relus?]: ",checkDeadRelu(activation["before_relu"]))
        
        #PASCAL CRITERIUM
        sOutputs, sTargets = scaleBackCoords(outputs, target, imsz)
        noTrue,noFalse,_ = pascalACC(sOutputs,sTargets)
        correct_count += noTrue
        false_count += noFalse
        
        
        loss = loss_fn(outputs,target) #L1 LOSS
        #loss = ops.generalized_box_iou_loss(output,target)
        loss.backward()
        running_loss += loss.item() #is complete EPOCHLOSS
        nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5) #experimentary
        model_opt.step()
        
       
        
        
    epochLoss = running_loss/counter 
    epochAcc = correct_count/len(trainloader.dataset)
    return epochLoss,correct_count,false_count,target,data["signal"],mask,epochAcc,model


def train_one_epoch_w_val(model,loss,train,oTrain,val_perc = 0.25,overfit=False,negative_print=False) -> float: 
    running_loss = 0.
    #src_mask = generate_square_subsequent_mask(32).to(device)
    correct_count = 0
    false_count = 0
    counter = 0
    
    
    if(overfit==True):
        train = oTrain
    else: 
        train = train #a bit clumsy, may be refactored
    
    torch.manual_seed(2)
    test_split,val_split = torch.utils.data.random_split(train,[math.ceil((1-val_perc)*len(train)),math.floor(val_perc*len(train))])
    
    trainloader = DataLoader(test_split,batch_size=BATCH_SZ,num_workers=0,shuffle=True)
    valloader = DataLoader(val_split,batch_size=BATCH_SZ,num_workers=0,shuffle=True)
    loss = 0
    
    model.train()
    for i, data in enumerate(trainloader):
        counter += 1
        model_opt.optimizer.zero_grad() #reset grads
        target = data["target"]
        mask = data["mask"]
        imsz = data["size"]
        
        #print("Mask:\n",data["mask"])
        #print("Input: \n",data["signal"])
        #print("Goal is: \n",data["target"])
        outputs = model(data["signal"],mask)
        
        
        #PASCAL CRITERIUM
        sOutputs, sTargets = scaleBackCoords(outputs, target, imsz) #with rescaling. Proved to be uneccesarry
        noTrue,noFalse,_ = pascalACC(sOutputs,sTargets)
        #print("\nScaled pascalACC returns:\n",pascalACC(sOutputs,sTargets))
        #print("\nOriginal pascalACC returns:\n",pascalACC(outputs,target))
        #print("\ngetIOU function returns:\n",getIOU(outputs,target,sensitivity=0.5))
        
        correct_count += noTrue
        false_count += noFalse
        
        
        loss = loss_fn(outputs,target) #SMOOTH L1
        #loss = ops.generalized_box_iou(outputs.to(dtype=torch.float64),target.to(dtype=torch.float64))
        loss.backward()
        running_loss += loss.item() #is complete EPOCHLOSS
        nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5) #experimentary
        model_opt.step()
        
    model.eval()
    val_counter = 0
    false_val_count = 0
    correct_val_count = 0
    val_loss = 0
    for i, data in enumerate(valloader):
        val_counter += 1
        target = data["target"]
        mask = data["mask"]
        outputs = model(data["signal"],mask)
        sOutputs, sTargets = scaleBackCoords(outputs, target, imsz)
        noTrue,noFalse,_ = pascalACC(sOutputs,sTargets)
        
        #PASCAL CRITERIUM
        #noTrue,noFalse,_ = getIOU(outputs,target,sensitivity=0.5)
        correct_val_count += noTrue
        false_val_count += noFalse
        
        
        loss = loss_fn(outputs,target) #L1 LOSS
        #loss = ops.generalized_box_iou_loss(outputs.to(dtype=torch.float32),target.to(dtype=torch.float32))
        
        val_loss += loss.item() #is complete EPOCHLOSS
        
       
        
    epochLoss = running_loss/counter 
    epochAcc = correct_count/len(trainloader)
    epochValLoss = val_loss/val_counter
    #print("complete over-epoch val loss: ",epochValLoss)   
    epochValAcc = correct_val_count/val_counter
    return epochLoss,correct_count,false_count,target,data["signal"],mask,epochAcc,model,epochValLoss,epochValAcc,trainloader,valloader



#def train_number_of_epochs(EPOCHS,model,loss,trainloader,oTrainLoader,overfit=False,negative_print=False):
epoch_number = 0
epochLoss = 0 
epochLossLI = []
epochAccLI = []
epochValLossLI = []
epochValAccLI = []
torch.autograd.set_detect_anomaly(True)

for epoch in (pbar:=tqdm(range(EPOCHS))):
    model.train(True)
    try:
        
        #epochLoss, correct_count, false_count,target,signal,mask,epochAcc,model = train_one_epoch(model,loss_fn,trainloader,oTrainLoader,overfit=OVERFIT,negative_print=False)
        epochLoss, correct_count, false_count,target,signal,mask,epochAcc,model,valLoss,valAcc,split_trainloader,split_valloader = train_one_epoch_w_val(model,loss_fn,train,overfitSet,overfit=OVERFIT,negative_print=False,val_perc=VAL_PERC)
        """#without tqdm
        #print("EPOCH {}:".format(epoch_number+1))    
        print("epoch loss {}".format(epochLoss))
        print("epoch val loss {}".format(valLoss))
        print("epoch val acc {}".format(valAcc))
        print("epoch train acc {}".format(epochAcc))
        """
        
        tmpStr = f" | avg train loss {epochLoss:.2f} | train acc: {epochAcc:.2f} | avg val loss: {valLoss:.2f} | avg val acc: {valAcc:.2f}"
        pbar.set_postfix_str(tmpStr)
        epochLossLI.append(epochLoss)
        epochAccLI.append(epochAcc)
        epochValAccLI.append(valAcc)
        epochValLossLI.append(valLoss)
        epoch_number += 1 
        #if(epochAcc==1):
         #   print("Perfect overfit at epoch {}".format(epoch_number+1))
        
    except KeyboardInterrupt:
        print("Manual early stopping triggered")
        break
    
 #       return epochLossLI,epochAccLI
    
 #epochLossLI,epochAccLI = train_number_of_epochs(EPOCHS,model,loss_fn,trainloader,oTrainLoader,overfit=OVERFIT,negative_print=False)

def save_epochs(loss,acc,classString,root_dir,mode):
    path = root_dir + classString
    if not os.path.exists(path):
        os.mkdir(path)
        
    torch.save(loss,path+"/"+classString+"_"+mode+"_losses.pth")
    torch.save(acc,path+"/"+classString+"_"+mode+"_acc.pth")
    return

save_epochs(epochLossLI,epochAccLI,classString,root_dir,mode="train")

    

def get_mean_model(trainloader): 
    """
    Batch-robust mean-calculator. 
        Input: dataloader structure of batched or unbatched input
        ---------------------------------------------------------
        Output: tensor, shape [1,4]
    """
    mean_vals = torch.zeros(1,4)
    for i, data in enumerate(trainloader): #get batch of training data
        for j in range(data["target"].shape[0]): #loop over batch-dimension
            mean_vals += data["target"][j]
    mean_vals /= len(trainloader.dataset)
    return mean_vals

    

    
def get_median_model(trainloader): #NEED TO FIX FOR BATCHES 
    """
    Batch-robust median-calculator. 
        Input: dataloader structure of batched or unbatched input
        ---------------------------------------------------------
        Output: tensor, shape [1,4]
    """
    holder_t = torch.zeros(len(trainloader.dataset),4)
    idx_space = 0
    for i, data in enumerate(trainloader):
        for j in range(data["target"].shape[0]):
            holder_t[j+idx_space] = data["target"][j]
        idx_space += j+1 #to ensure support for random and non-equal batch-sizes
    median_t,_ = torch.median(holder_t,dim=0,keepdim=True)
    #if want debug: return t_holder
    return median_t
    
#h1.remove()
h2.remove()   
h3.remove()
h4.remove()
h5.remove()
h6.remove()




#---------------------TEST AND EVAL -------------#
#1. TEST-LOOP ON TRAIN-SET
trainsettestLosses = []

#eval on TRAIN SET 
meanModel = get_mean_model(split_trainloader)
medianModel = get_median_model(split_trainloader)

model.train(False)
if(OVERFIT):
    print("Evaluating overfit on first {} instances".format(len(split_trainloader)))
    
    no_overfit_correct = 0
    no_overfit_false = 0
    no_mean_correct = 0
    no_mean_false = 0
    no_med_correct = 0
    no_med_false = 0
    overfit_save_struct = []
    
    for i, data in enumerate(split_trainloader):
        signal = data["signal"]
        target = data["target"]
        mask = data["mask"]
        size = data["size"]
        name = data["file"]
        output = model(signal,mask)
        batchloss = loss_fn(target,output) #L1 LOSS
       # batchloss = ops.generalized_box_iou_loss(output.to(dtype=torch.float32),target.to(dtype=torch.float32))
        
        accScores = pascalACC(output,target)
        no_overfit_correct += accScores[0]
        no_overfit_false += accScores[1]
        IOU = accScores[2]
        for i in range(len(IOU)):
            if(IOU[i]>0.5):
                overfit_save_struct.append([name,str(1),IOU[i],target,output,size]) #filename, pred-status: correct(1):false(0), IOU-value, ground-truth, prediction-value 
            else:
                overfit_save_struct.append([name,str(0),IOU[i],target,output,size])
        print("Filename: {}\n Target: {}\n Prediction: {}\n Loss: {}\n IOU: {}".format(data["file"],data["target"],output,batchloss,IOU))
        trainsettestLosses.append(batchloss)
        
        #fix meanModel to have as many entrys as target-tensor: 
        n_in_batch = target.shape[0]
        meanModel_tmp = meanModel.repeat(n_in_batch,1) #make n_in_batch copies along batch-dimension
        medianModel_tmp = medianModel.repeat(n_in_batch,1)
        
        accScores = pascalACC(meanModel_tmp,target)
        no_mean_correct += accScores[0]
        no_mean_false += accScores[1]
        
        accScores = pascalACC(medianModel_tmp,target)
        no_med_correct += accScores[0]
        no_med_false += accScores[1]
        
    print("---------------------------EVAL on ALL {} overfit-train-images---------------------------".format(len(split_trainloader)))    
    print("\nTransformer accuracy with PASCAL-criterium on overfit set: {}/{}, percentage: {}".format(no_overfit_correct,no_overfit_false+no_overfit_correct,no_overfit_correct/(no_overfit_false+no_overfit_correct)))    
    print("\nMean model accuracy with PASCAL-criterium on overfit set: {}/{}, percentage: {}".format(no_mean_correct,no_mean_false+no_mean_correct,no_mean_correct/(no_mean_false+no_mean_correct)))
    print("\nMedian model accuracy with PASCAL-criterium on overfit set: {}/{}, percentage: {}".format(no_med_correct,no_med_false+no_med_correct,no_med_correct/(no_med_false+no_med_correct)))
    torch.save(overfit_save_struct,root_dir+classString+"/"+classString+"_"+"test_on_train_results.pth")
    print("\n   Results saved to file: ",root_dir+classString+"/"+classString+"_"+"test_on_train_results.pth")
    
    
else:
    print("SECOND LOOP Evaluating on first {} instances".format(len(split_trainloader)))
    
    no_train_correct = 0
    no_train_false = 0
    no_mean_correct = 0
    no_mean_false = 0
    no_med_correct = 0
    no_med_false = 0
    train_save_struct = []
    
    with torch.no_grad():
        for i, data in enumerate(split_trainloader):
            signal = data["signal"]
            target = data["target"]
            mask = data["mask"]
            size = data["size"]
            name = data["file"]
            output = model(signal,mask)
            batchloss = loss_fn(target,output) #L1-loss
           # batchloss = ops.generalized_box_iou_loss(output.to(dtype=torch.float32),target.to(dtype=torch.float32))
            accScores = pascalACC(output,target)
            no_train_correct += accScores[0]
            no_train_false += accScores[1]
            IOU = accScores[2]
            for i in range(len(IOU)):
                if(IOU[i]>0.5):
                    train_save_struct.append([name,str(1),IOU[i],target,output,size]) #filename, pred-status: correct(1):false(0), IOU-value, ground-truth, prediction-value 
                else:
                    train_save_struct.append([name,str(0),IOU[i],target,output,size])
            print("Filename: {}\n Target: {}\n Prediction: {}\n Loss: {}\n".format(data["file"],data["target"],output,batchloss))
            trainsettestLosses.append(batchloss)
            
            #print("IOU: ",IOU)
            n_in_batch = target.shape[0]
            meanModel_tmp = meanModel.repeat(n_in_batch,1) #make n_in_batch copies along batch-dimension
            medianModel_tmp = medianModel.repeat(n_in_batch,1) #make n_in_batch copies along batch-dimension
            
            accScores = pascalACC(meanModel_tmp,target)
            no_mean_correct += accScores[0]
            no_mean_false += accScores[1]
            
            accScores = pascalACC(medianModel_tmp,target)
            no_med_correct += accScores[0]
            no_med_false += accScores[1]
            
                
            
    print("---------------EVAL on ALL {} training-images---------------".format(len(split_trainloader)))
    print("\n\nTransformer accuracy with PASCAL-criterium on training-data used: {}/{}, percentage: {}".format(no_train_correct,no_train_false+no_train_correct,no_train_correct/(no_train_false+no_train_correct)))    
    print("\nMean model accuracy with PASCAL-criterium on overfit set: {}/{}, percentage: {}".format(no_mean_correct,no_mean_false+no_mean_correct,no_mean_correct/(no_mean_false+no_mean_correct)))
    print("\nMedian model accuracy with PASCAL-criterium on overfit set: {}/{}, percentage: {}".format(no_med_correct,no_med_false+no_med_correct,no_med_correct/(no_med_false+no_med_correct)))
    torch.save(train_save_struct,root_dir+classString+"/"+classString+"_"+"test_on_train_results.pth")
    print("\n   Results saved to file: ",root_dir+classString+"/"+classString+"_"+"test_on_train_results.pth")
    



#2. TEST-LOOP ON TEST-SET
no_test_correct = 0 
no_test_false = 0
no_test_mean_correct = 0 
no_test_mean_false = 0
no_test_median_correct = 0
no_test_median_false = 0
testlosses = []
correct_false_list = []

# meanModel = get_mean_model(oTrainLoader)
# medianModel = get_median_model(oTrainLoader)

with torch.no_grad():
    running_loss = 0 
    for i, data in enumerate(testloader):
        signal = data["signal"]
        target = data["target"]
        mask = data["mask"]
        name = data["file"]
        size = data["size"]
        output = model(signal,mask)
        batchloss = loss_fn(target,output) #L1 Loss
        #batchloss = ops.generalized_box_iou_loss(output.to(dtype=torch.float32),target.to(dtype=torch.float32))
        running_loss += batchloss.item()
        testlosses.append(batchloss.item())
        accScores = pascalACC(output,target)
        IOU = accScores[2]
        for i in range(len(IOU)):
            if(IOU[i]>0.5):
                correct_false_list.append([name,str(1),IOU[i],target,output,size]) #filename, pred-status: correct(1):false(0), IOU-value, ground-truth, prediction-value 
            else:
                correct_false_list.append([name,str(0),IOU[i],target,output,size])
            
        
        no_test_correct += accScores[0]        
        no_test_false += accScores[1]
        
        n_in_batch = target.shape[0]
        meanModel_tmp = meanModel.repeat(n_in_batch,1) #make n_in_batch copies along batch-dimension
        medianModel_tmp = medianModel.repeat(n_in_batch,1)
        accScores = pascalACC(meanModel_tmp,target)
        
        no_test_mean_correct += accScores[0]
        no_test_mean_false += accScores[1]
        
        accScores = pascalACC(medianModel_tmp,target)
        no_test_median_correct += accScores[0]
        no_test_median_false += accScores[1]
        
        if i!=0 and i%100==0:
            print("L1-loss on every over batch {}:{}: {}\n".format(i-100,i,running_loss/100))
            running_loss = 0 
            


testmeanAcc = no_test_mean_correct/(no_test_mean_false+no_test_mean_correct)
testmedianAcc = no_test_median_correct/(no_test_median_false+no_test_median_correct)

print("---------------EVAL on ALL {} test-images---------------".format(len(testloader)))
print("\nTransformer accuracy with PASCAL-criterium: {}/{}, percentage: {}".format(no_test_correct,no_test_false+no_test_correct,no_test_correct/(no_test_false+no_test_correct)))    
print("\nMean model accuracy with PASCAL-criterium: {}/{}, percentage: {}".format(no_test_mean_correct,no_test_mean_false+no_test_mean_correct,testmeanAcc))    
print("\nMedian model accuracy with PASCAL-criterium: {}/{}, percentage: {}".format(no_test_median_correct,no_test_median_false+no_test_median_correct,testmedianAcc))    
torch.save(correct_false_list,root_dir+classString+"/"+classString+"_"+"test_on_test_results.pth")
print("\n   Results saved to file: ",root_dir+classString+"/"+classString+"_"+"test_on_test_results.pth")
    




print("\n......Entering plotting module......\n")

def write_fig(root_dir,classString,pltobj,title=None,mode=None):
    path = root_dir + classString + "/graphs/"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created graph-dir: ",path)
    if title==None:
        import time
        clock = time.localtime()
        timeString = str(clock[0])+"_"+str(clock[2])+"_"+str(clock[1]) + "_time_"+str(clock[3])+"-"+str(clock[4])+"-"+str(clock[5])
        title = classString+"_"+mode+"_"+timeString+".png"
        del timeString
    pltobj.savefig(path+title)
    print("Saved figure to ",path+title)
    return None

    


import matplotlib.pyplot as plt
plt.figure(1337)
plt.plot(epochLossLI)
plt.plot(epochValLossLI)
if EPOCHS > 500:
    xticks = range(0,len(epochLossLI),100)
elif EPOCHS >100:
    xticks = range(0,len(epochLossLI),10)
else:
    xticks = range(0,len(epochLossLI))
    
plt.xticks(xticks)
plt.ylabel("Smooth L1-loss, beta=1") 
plt.xlabel("Epoch")
plt.legend(["Training loss","Validation loss"])
if not OVERFIT:
    plt.suptitle("Transformer training error {} images, validation {}".format(math.floor(len(train)*(1-VAL_PERC)),math.ceil(len(train)*VAL_PERC)))
    if SAVEFIGS:
        write_fig(root_dir,classString,plt,title="learning_curve",mode="train")
    
if(OVERFIT):
    plt.title("Train-error on constant subset of {} training images, {} val images".format(math.floor(NUM_IN_OVERFIT*(1-VAL_PERC)),math.ceil(NUM_IN_OVERFIT*VAL_PERC)))
    if SAVEFIGS:
        write_fig(root_dir,classString,plt,title="{}_overfitset".format(len(overfitSet)),mode="overfit")
    

    
    

