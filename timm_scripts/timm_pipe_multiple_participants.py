#import timm
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
    def __init__(self,pascalobj,root_dir,classes,sTransform=None,imTransform = None,coordTransform = None,num_participants=1):
        self.ABDset,self.filenames,self.eyeData,self.targets,self.classlabels,self.imdims,self.length = generate_DS(pascalobj,classes)
        self.root_dir = root_dir
        self.sTransform = sTransform
        self.imTransform = imTransform
        self.coordTransform = coordTransform
        self.deep_representation = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ViT_model = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=0, global_pool='').to(device)
        self.NUM_PARTICIPANTS = num_participants
        
    def __len__(self): 
        return len(self.ABDset)
    
    
    def __getitem__(self,idx): #for images
        if torch.is_tensor(idx):
            idx = idx.tolist() #for indexing
        
        
        filename = self.filenames[idx]
        image = Image.open(self.root_dir + filename)
        
       
        if(self.NUM_PARTICIPANTS==1):
            signals = self.eyeData[idx][0] #participant zero only
        
        else: 
            signals = self.eyeData[idx,:self.NUM_PARTICIPANTS]
        

        
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
            
        with torch.no_grad():    
            sample["image"] = self.ViT_model.forward_features(sample["image"])
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
        if x.ndim==2: #only one participant
            xtmp = torch.zeros(32,2)
            numel = x.shape[0]
            xtmp[:numel,:] = x[:,:]
            maskTensor = (xtmp[:,0]==0)    
        else: #multiple participants
            xtmp = torch.zeros(x.shape[0],32,2)
            numel = x.shape[1]
            xtmp[:,:numel,:] = x[:,:,:]
            maskTensor = (xtmp[:,:,0]==0)    
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
        
        if(eyeCoords.ndim==2): #single participant 
            outCoords = torch.ones(32,2)
            #old ineccesary #eyeCoords[inMask] = float('nan') #set nan for next calculation -> division gives nan. Actually ineccesary
            #handle masking
            #inMask = (eyeCoords==0.0) #mask-value
            #calculate scaling
            outCoords[:,0] = eyeCoords[:,0]/imdims[1]
            outCoords[:,1] = eyeCoords[:,1]/imdims[0]
            
            #reapply mask
            outCoords[mask] = 0.0 #reapply mask
            
        if(eyeCoords.ndim==3): #multiple participants
            num = eyeCoords.shape[0]
            outCoords = torch.ones(num,32,2)
            outCoords[:,:,0] = eyeCoords[:,:,0]/imdims[1]
            outCoords[:,:,1] = eyeCoords[:,:,1]/imdims[0]
            
            #reapply mask
            outCoords[mask] = 0.0 #reapply mask
        
        
        #scale bounding-box
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
BATCH_SZ = 1
NLAYERS = 1
NHEADS = 1
EPOCHS = 3
NUM_IN_OVERFIT = 2
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
    
    fullDataset = PASCALdataset(dataFrame, root_dir, classesOC,sTransform=SignalTrans,imTransform = ImTrans,coordTransform = CoordTrans,num_participants=2) #init dataset as torch.Dataset.
    
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
    valSet = torch.utils.data.Subset(train,valIDX[:6])
    trainloader = DataLoader(overfitSet,batch_size=BATCH_SZ,shuffle=True,num_workers=0,generator=g)
    print("Overwrote trainloader with overfit-set of length {}".format(ofIDX.shape[0]))
    valloader = DataLoader(valSet,batch_size=BATCH_SZ,shuffle=True,num_workers=0,generator=g)
    print("Valloader of constant length {}".format(valIDX.shape[0]))


# for i in range(1): #get first sample for debugging
#     data = next(iter(trainloader))
#     signal = data["signal"]
#     mask = data["mask"]
#     output = data["image"].squeeze(1) #ViT output. Now has dimensions [batch_sz,197,768]. Check for batches aswell.
#     im = data["orgimage"] #normalized and reshaped squared original image

    
# ------------------------------------FOR UNBATCHED INPUT FOR DEBUGGING. ALL FUNCTIONS ONLY TAKE FIRST IMAGE IN BATCH
# #SHOW SQUARE-RESHAPED IMG 
# plt.figure(0)    
# plt.imshow(im[0].squeeze(0).permute(1,2,0))
# plt.title("Image used in ViT")


# #SAMPLE SQUARE BACK TO ORG SIZE
# plt.figure(1)
# size = (int(data["size"][0,1].item()),int(data["size"][0,0].item())) #get original image shape. Remember: DS is in H,W and resize uses W,H
# sampled_back_image = torchvision.transforms.ToPILImage()(im[0].squeeze(0)) #convert to PIL-format.
# sampled_back_image = sampled_back_image.resize(size,Image.BILINEAR) #reshape back to org size
# plt.imshow(sampled_back_image)
# plt.title("Resampled image")
    
   
# # output_no_cls = output[0,1:,:]
def get_deep_seq(sample,get_feature_map=False):
    """
    Function which takes an eye signal for an image, and finds the corresponding deep-feature-values in the feature_map
    Concatenates x,y as first deep-feature-dims, deletes CLS-tokens
    Parameters
    ----------
    sample: 
        A datasample from dataloader containing:
            signal : torch int tensor
                non-normalized (x,y)-sequence of fixations. Holds coordinates. Shape: [batch_sz,32,2]
            mask : torch bool tensor
                simple tensor, which states which values in signal deep features should be extracted from. Shape: [batch_sz,32]
            featuremap : torch float tensor
                ViT-extracted feature-map of image of interest. 
    get_feature_map : boolean
        If set to true, returns a list of all constructed featuremaps. Only used for debugging (check_correctness_batch_interp) and possibly later visualizations                
    Returns
    -------
    holder_t: 
        A batch of deep-feature-representations with dimensionality [batch_sz, seq_len, 2+768]. This is input to transformer architecture. 2+ is x,y. 
    featuremap_holder_l: a list
    """
    MULTIPLE = False
    signal = sample["signal"].to(device)
    if signal.ndim==2: #fix single batches which come without batch. Should not be an issue using trainloader
        signal = signal.unsqueeze(0)
    if signal.ndim==4: #case: multiple participants [bs,participantNo,32,2]
        NUM = signal.shape[1]
        MULTIPLE = True
    BSZ = signal.shape[0]
    featuremap = sample["image"].to(device)
    size = sample["size"].to(device)
    CLS = featuremap[:,:,0,:] #pull out cls-tokens. Gets all batches, size [batch_sz,1,1,768]
    featuremap_noCLS = featuremap[:,:,1:,:]
    featuremap_noCLS = featuremap_noCLS.view(BSZ,14,14,768)
    featuremap_perm = featuremap_noCLS.permute(0,-1,1,2) #bilinear interp runs over last two dimensions. Therefore permute to [batch_sz,768,x,y]
    if(MULTIPLE==False):
        holder_t = torch.zeros(BSZ,signal.shape[1],featuremap.shape[-1]+2).to(device) #[BS,32,770]
    if(MULTIPLE==True):
        holder_t = torch.zeros(BSZ,NUM,signal.shape[2],featuremap.shape[-1]+2).to(device) #[BS,NUM_PARTS,32,770]
    featuremap_holder_l = []
    for i in range(BSZ):
        if(MULTIPLE==False):
            sizes = (int(size[i][1].item()),int(size[i][0].item())) #x,y. Remember data["batch"] is h,w
            featuremap_intpl = F.interpolate(featuremap_perm[i].unsqueeze(0),sizes,mode='bilinear') #interpolate tensor. Sadly cant be done outside of loop, as sizes to interpolate to have to be constant (cant be tensor)
            featuremap_intpl = featuremap_intpl.permute(0,2,3,1) #get back to format [batch_sz,x,y,embed_dim]
            featuremap_holder_l.append(featuremap_intpl)
            #now get indices by slicing 
            x = signal[i,:,0].to(dtype=torch.long)
            y = signal[i,:,1].to(dtype=torch.long)
            holder_slice = featuremap_intpl[(slice(None),x,y,slice(None))] #returns sliced featuremap. Format: [1,(seq_len),768]
          #   print(holder_t.shape)
            holder_t[i,:,2:] = holder_slice
        if(MULTIPLE==True):
            for j in range(NUM):
                sizes = (int(size[i][1].item()),int(size[i][0].item())) #x,y. Remember data["batch"] is h,w
                featuremap_intpl = F.interpolate(featuremap_perm[i].unsqueeze(0),sizes,mode='bilinear') #interpolate tensor. Sadly cant be done outside of loop, as sizes to interpolate to have to be constant (cant be tensor)
                featuremap_intpl = featuremap_intpl.permute(0,2,3,1) #get back to format [batch_sz,x,y,embed_dim]
                featuremap_holder_l.append(featuremap_intpl)
                #now get indices by slicing 
                x = signal[i,j,:,0].to(dtype=torch.long)
                y = signal[i,j,:,1].to(dtype=torch.long)
                holder_slice = featuremap_intpl[(slice(None),x,y,slice(None))] #returns sliced featuremap. Format: [1,(seq_len),768]
              #   print(holder_t.shape)
                holder_t[i,j,:,2:] = holder_slice
    if(MULTIPLE==False):
        holder_t[:,:,0] = sample["scaled_signal"][:,:,0]
        holder_t[:,:,1] = sample["scaled_signal"][:,:,1]
    if(MULTIPLE==True):
        holder_t[:,:,:,0] = sample["scaled_signal"][:,:,:,0]
        holder_t[:,:,:,1] = sample["scaled_signal"][:,:,:,1]
    
    if(get_feature_map==True):
        return holder_t,featuremap_holder_l
    else:
        return holder_t

#deep_seq = get_deep_seq(data,get_feature_map=False)

# def check_correctness_batch_interp(featuremap,deep_seq,signal,mask,printVals = False):
#     """
#     Simple function for debugging get_deep_seq. Prints signal-values from featuremap and from extracted sequence.
#     Ensure that they are the same
#     Parameters
#     ----------
#     featuremap : list containing torch tensors
#         List of complete featuremaps for the batched sequence. Format: list of length BATCH_SZ containing tensors of shape [1,im_width,im_height,embedding dimension] with coordinates [x,y]
#     deep_seq : torch tensor, [batch_size,32,768]
#         Latent space representation of image-fixations.
#     signal : torch tensor, [batch_size,32,2]
#         Contains image-fixation coordinates
#     mask : torch boolean tensor
#         Describes if entry in signal has been zero-padded. False for non-padded elements, true for padded elements 
    
#     printVals: boolean 
#         Prints values to screen if set to true (only used for debugging if exit code is failure)
    
#     Returns
#     -------
#     integer value exit-code 
#           0: if all elements are the same 
#         -1: if some elements are not the same 
#     """
#     BATCH_SZ = signal.shape[0]
#     rev_mask = ~mask
#     rev_mask = rev_mask.to(dtype=torch.long)
#     num_entries = torch.sum(rev_mask,1) #get number of entries in batchwise signal
#     for batch in range(BATCH_SZ):
#         for entry in range(num_entries[batch].item()):
#             get_x_coord = int(signal[batch,entry,0].item())  
#             get_y_coord = int(signal[batch,entry,1].item())
#             if(printVals==True):
#                 print("\nChecking coordinates: ({},{}) in image {}".format(get_x_coord,get_y_coord,batch))
#             for channel in [0,10,15,200,767]: #random sample including last index 
#                 featureVal = featuremap[batch][0,get_x_coord,get_y_coord,channel]
#                 deepVal = deep_seq[batch,entry+1,channel+2] #ignore CLS-tokens in all cases, and remember xy-offset
#                 if(printVals==True):
#                     print("\nFeaturemap-entry for coordinates, channel {}:{}".format(channel,featureVal))
#                     print("\nExtracted deep_seq entry for corresponding coordinate, channel {}:{}".format(channel,deepVal))
#                 if not (featureVal==deepVal):
#                     return -1 #exit with failure
#     return 0 #exit with success
                
# #check_correctness_batch_interp(featuremaps,deep_seq,signal,mask,printVals=False)            


# #featuresNP = featuremaps[0].squeeze(0).detach().cpu().numpy()
# #featuremap0 = featuresNP[:,:,0].T #plt wants transpose. Your handling is correct
# #featuresIM = np.array(Image.fromarray(featuremap0)) #zeroth channel for checking
# #plt.figure(123123)
# #plt.imshow(featuresIM)


model = vm.eyeFormer_ViT(dropout=DROPOUT,n_layers=NLAYERS,num_heads=NHEADS,num_participants=2).to(device)
#model(deep_seq,mask)
model_opt = vm.NoamOpt(model.d_model,LR_FACTOR,NUM_WARMUP,torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
loss_fn = nn.SmoothL1Loss(beta=BETA) #default: mean and beta=1.0

#for i, data in enumerate(trainloader):
#    if(i==3):
#        break
#    sample = data


def train_one_epoch(model,loss,trainloader,negative_print=False) -> float: 
    
    running_loss = 0.
    #src_mask = generate_square_subsequent_mask(32).to(device)
    correct_count = 0
    false_count = 0
    counter = 0
    
    tIOUli_holder = []
    
    for i, data in enumerate(trainloader):
        if i>0:
            break
        counter += 1
        model_opt.optimizer.zero_grad() #reset grads
        target = data["target"].to(device)
        mask = data["mask"].to(device)
        signal = data["signal"].to(device)
        dSignal = get_deep_seq(data)
        #print("Mask:\n",data["mask"])
        #print("Input: \n",data["signal"])
        #print("Goal is: \n",data["target"])
        outputs = model(dSignal,mask)
        
        #PASCAL CRITERIUM
        noTrue,noFalse,IOU_li = vm.pascalACC(outputs,target)
        tIOUli_holder = tIOUli_holder + IOU_li
        correct_count += noTrue
        false_count += noFalse
        
        loss = loss_fn(outputs,target) #L1 LOSS
        #loss = ops.generalized_box_iou_loss(output,target)
        loss.backward()
        running_loss += loss.item() #is complete EPOCHLOSS
        nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5) #experimentary
        model_opt.step()
        
    IOU_mt = sum(tIOUli_holder)/len(tIOUli_holder)    
    epochLoss = running_loss/len(trainloader.dataset) 
    epochAcc = correct_count/len(trainloader.dataset) #TP / (complete trainingset length)    
    
    return epochLoss,correct_count,false_count,target,data["signal"],mask,epochAcc,model,IOU_mt


def train_one_epoch_w_val(model,loss,trainloader,valloader,negative_print=False,DEBUG=False) -> float: 
    running_loss = 0.
    #src_mask = generate_square_subsequent_mask(32).to(device)
    correct_count = 0
    false_count = 0
    counter = 0
    
    if(DEBUG):
        train__L = []
        val__L = []
        for i, data in enumerate(trainloader):
            train__L.append(data["file"][0])
        for i, data in enumerate(valloader):
            val__L.append(data["file"][0])
        
        train_s = set(train__L) 
        val_s = set(val__L)
        s = train_s & val_s 
        if len(s)!=0:
            print("!!!!WARNING: VALIDATION SET DOES CONTAIN TRAINING DATA!!!")
    
    #VALIDATION FIRST
    vIOUli_holder = []
    false_val_count = 0
    correct_val_count = 0
    val_loss = 0
    counter = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valloader):
            if i>0: 
                break
            counter += 1 
            mask = data["mask"]
            signal = data["signal"]
            dSignal = get_deep_seq(data)
            outputs = model(dSignal,mask)
            target = data["scaled_target"]
            outputs_boxed = center_to_box(outputs)
            noTrue,noFalse,IOUli_v = vm.pascalACC(outputs_boxed,target)
            
            vIOUli_holder = vIOUli_holder+IOUli_v 
            #PASCAL CRITERIUM
            #noTrue,noFalse,_ = getIOU(outputs,target,sensitivity=0.5)
            correct_val_count += noTrue
            false_val_count += noFalse
            
            
            loss = loss_fn(outputs_boxed,target) #L1 LOSS. Mode: "mean"
            #loss = ops.generalized_box_iou_loss(outputs.to(dtype=torch.float32),target.to(dtype=torch.float32))
            
            val_loss += loss.item() #is complete EPOCHLOSS
    
    loss = 0
    
    model.train()
    tIOUli_holder = []
    for i, data in enumerate(trainloader):
        if i>3: 
            break
        counter += 1
        model_opt.optimizer.zero_grad() #reset grads
        mask = data["mask"]
        dSignal = get_deep_seq(data)
        
        #print("Mask:\n",data["mask"])
        #print("Input: \n",data["signal"])
        #print("Goal is: \n",data["target"])
        outputs = model(dSignal,mask)
        target = data["scaled_target"]
        outputs_boxed = center_to_box(outputs)
        noTrue,noFalse,IOUli_v = vm.pascalACC(outputs_boxed,target)
        
        
        #PASCAL CRITERIUM
        #sOutputs, sTargets = scaleBackCoords(outputs, target, imsz) #with rescaling. Proved to be uneccesarry
        noTrue,noFalse,IOUli_t = vm.pascalACC(outputs,target)
        #print("\nScaled pascalACC returns:\n",pascalACC(sOutputs,sTargets))
        #print("\nOriginal pascalACC returns:\n",pascalACC(outputs,target))
        #print("\ngetIOU function returns:\n",getIOU(outputs,target,sensitivity=0.5))
        tIOUli_holder = tIOUli_holder+IOUli_t 
        
        correct_count += noTrue
        false_count += noFalse
        
        
        loss = loss_fn(outputs_boxed,target) #SMOOTH L1. Mode: mean
        #loss = ops.generalized_box_iou(outputs.to(dtype=torch.float64),target.to(dtype=torch.float64))
        running_loss += loss.item() #is complete EPOCHLOSS
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5) #experimentary
        model_opt.step()
    
    if(DEBUG==True):
        print("t holder becomes: \n",tIOUli_holder)
        print("\nsum becomes",sum(tIOUli_holder))
        print("\nLen becomes",len(tIOUli_holder))
    IOU_mt = sum(tIOUli_holder)/len(tIOUli_holder)        
    
    
        
       
    if(DEBUG==True):
        print("v holder becomes: \n",vIOUli_holder)
        print("\nsum becomes",sum(vIOUli_holder))
        print("\nLen becomes{}".format(len(vIOUli_holder)))
    IOU_mv = sum(vIOUli_holder)/len(vIOUli_holder)
    
    epochLoss = running_loss/len(trainloader.dataset) 
    epochAcc = correct_count/len(trainloader.dataset) #TP / (complete trainingset length)
    #print("correct count is: ",correct_count)
    #print("trainloader dset len is: ",len(trainloader.dataset))
    epochValLoss = val_loss/len(valloader.dataset) 
    #print("complete over-epoch val loss: ",epochValLoss)   
    epochValAcc = correct_val_count/len(valloader.dataset)
    return epochLoss,correct_count,false_count,target,data["signal"],mask,epochAcc,model,epochValLoss,epochValAcc, IOU_mt, IOU_mv,data,dSignal

print("-----------------------------------------------------------------------------")
if(OVERFIT):
    print("Model parameters:\n tL: {}\n vL: {}\nlrf: {}\n num_warmup: {}\n Dropout: {}\n Beta: {}\n NHEADS: {}\n NLAYERS: {}".format(NUM_IN_OVERFIT,len(valIDX),LR_FACTOR,NUM_WARMUP,DROPOUT,BETA,NHEADS,NLAYERS))
else:
    print("Model parameters:\n tL: {}\n \nlrf: {}\n num_warmup: {}\n Dropout: {}\n Beta: {}\n NHEADS: {}\n NLAYERS: {}".format(NUM_IN_OVERFIT,LR_FACTOR,NUM_WARMUP,DROPOUT,BETA,NHEADS,NLAYERS))
    
#def train_number_of_epochs(EPOCHS,model,loss,trainloader,oTrainLoader,overfit=False,negative_print=False):
epoch_number = 0
epochLoss = 0 
epochLossLI = []
epochAccLI = []
epochValLossLI = []
epochValAccLI = []
epochIOU = []
trainIOU = []
valIOU = []
torch.autograd.set_detect_anomaly(True)

for epoch in (pbar:=tqdm(range(EPOCHS))):
    try:        
        if(OVERFIT):
            epochLoss, correct_count, false_count,target,signal,mask,epochAcc,model,valLoss,valAcc,IOU_t,IOU_v,trfdata,dSignal = train_one_epoch_w_val(model,loss_fn,trainloader,valloader,negative_print=False,DEBUG=True)
            tmpStr = f" | avg train loss {epochLoss:.4f} | train acc: {epochAcc:.4f} | avg val loss: {valLoss:.4f} | avg val acc: {valAcc:.4f} | Mean Train IOU: {IOU_t:.4f} | Mean Val IOU: {IOU_v:.4f} |"
            epochValAccLI.append(valAcc)
            epochValLossLI.append(valLoss)
            valIOU.append(IOU_v)
            
        else:
            epochLoss, correct_count, false_count,target,signal,mask,epochAcc,model,IOU_t = train_one_epoch(model,loss_fn,trainloader,negative_print=False)
            tmpStr = f" | avg train loss {epochLoss:.2f} | train acc: {epochAcc:.2f} | Epoch train IOU {IOU_t:.2f} |"
            
        pbar.set_postfix_str(tmpStr)
        epochLossLI.append(epochLoss)
        epochAccLI.append(epochAcc)
        trainIOU.append(IOU_t)
        epoch_number += 1 
        
        
    except KeyboardInterrupt:
        print("Manual early stopping triggered")
        break
    


def save_split(trainloader,valloader,classString,root_dir,params):
    """
    Simple function for saving filenames of images used in train- and val-split respectively for ensuring reproducibility
    Args: 
        trainloader (pyTorch dataloader)
        valloader (pyTorch dataloader)
        classString: string, name descriptor of class of investigation
        root_dir: usually os.path.dirname(__file__)
        
    Returns: 
        None
        
    """
    path = root_dir + "/" + classString + "/nL_" + str(params[0]) +"_nH_" + str(params[1])+"/"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created dir in: ",path)
    trainlist = []
    vallist = []
    for i,data in enumerate(trainloader):
        trainlist.append(data["file"][0])
    for i, data in enumerate(valloader):
        vallist.append(data["file"][0])
    s = path+classString+"_train_split_len_"+str(len(trainloader.dataset))+".pth"
    torch.save(trainlist,s)
    s = path+classString+"_val_split_len_"+str(len(valloader.dataset))+".pth"
    torch.save(vallist,s)
    

def save_epochs(loss,acc,classString,root_dir,mode,params):
    path = root_dir + "/"+classString + "/"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created dir in: ",path)
    path += "nL_" + str(params[0]) +"_nH_" + str(params[1]) +"/"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created subdir in: ",path)
    
    if(mode=="eval"):
        path += "eval/"
        if not os.path.exists(path):
            os.mkdir(path)
            print("Created subdir in: ",path)
        torch.save(loss,path+classString+"_"+mode+"_losses.pth")
        print("Saved loss results to: ",path+classString+"_"+mode+"_losses.pth")
        torch.save(acc,path+"/"+classString+"_"+mode+"_acc.pth")
        print("Saved accuracy results to: ",path+classString+"_"+mode+"_acc.pth")
    else:
        torch.save(loss,path+classString+"_"+mode+"_losses.pth")
        print("Saved loss results to: ",path+classString+"_"+mode+"_losses.pth")
        torch.save(acc,path+"/"+classString+"_"+mode+"_acc.pth")
        print("Saved accuracy results to: ",path+classString+"_"+mode+"_acc.pth")
        
    return

def save_IOU(IOU_li,classString,root_dir,params,mode):
    path = root_dir + "/"+classString+"/"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created dir in: ",path)
    path += "nL_" + str(params[0]) +"_nH_" + str(params[1]) +"/"+mode+"/"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created subdir in: ",path)
        
    torch.save(IOU_li,path+"epochIOU.pth")
    print("Wrote epoch IOU's to scratch in :",path)
    return None

def save_model(model,classString,root_dir,params,mode):
    path = root_dir + "/" + classString+"/"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created dir in: ",path)
    path += "nL_" + str(params[0]) +"_nH_" + str(params[1]) +"/"+mode+"/"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created subdir in: ",path)
    torch.save(model.state_dict(),path+"model.pth")
    print("Wrote finished model to scratch in :",path)
    return None

if(EVAL==0):
    save_epochs(epochLossLI,epochAccLI,classString,timm_root,mode="result",params=[NLAYERS,NHEADS])
    save_IOU(trainIOU,classString,timm_root,params=[NLAYERS,NHEADS],mode="result")
    save_model(model,classString,timm_root,params=[NLAYERS,NHEADS],mode="result")

if(EVAL==1):
    save_epochs(epochLossLI,epochAccLI,classString,timm_root,mode="eval",params=[NLAYERS,NHEADS])
    save_IOU(trainIOU,classString,timm_root,params=[NLAYERS,NHEADS],mode="eval")
    save_model(model,classString,timm_root,params=[NLAYERS,NHEADS],mode="eval")
    

if(OVERFIT==True and EVAL==0):
    save_epochs(epochValLossLI,epochValAccLI,classString,timm_root,mode="val",params=[NLAYERS,NHEADS])
    save_split(trainloader,valloader,classString,timm_root,params=[NLAYERS,NHEADS])
    print("\nWrote train-val-split to scratch.\n")
    

def get_mean_model(trainloader): 
    """
    Batch-robust mean-calculator. 
        Input: dataloader structure of batched or unbatched input
        ---------------------------------------------------------
        Output: tensor, shape [1,4]
    """
    mean_vals = torch.zeros(1,4)
    for i, data in enumerate(trainloader): #get batch of training data
        for j in range(data["scaled_target"].shape[0]): #loop over batch-dimension
            mean_vals += data["scaled_target"][j]
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
    



# #---------------------TEST AND EVAL -------------#
paramsString = timm_root +"/"+ classString + "/nL_" + str(NLAYERS) +"_nH_" + str(NHEADS)+"/" #for saving to correct dirs

#1. TEST-LOOP ON TRAIN-SET
trainsettestLosses = []
IOU_tr_li = []

#eval on TRAIN SET 
meanModel = get_mean_model(trainloader)
medianModel = get_median_model(trainloader)

model.eval()

print("Entered evaluation-phase.")
if(OVERFIT):
    print("Model parameters:\n tL: {}\n vL: {}\nlrf: {}\n num_warmup: {}\n Dropout: {}\n Beta: {}\n NHEADS: {}\n NLAYERS: {}".format(NUM_IN_OVERFIT,len(valIDX),LR_FACTOR,NUM_WARMUP,DROPOUT,BETA,NHEADS,NLAYERS))
else:
    print("Model parameters:\n tL: {} \nlrf: {}\n num_warmup: {}\n Dropout: {}\n Beta: {}\n NHEADS: {}\n NLAYERS: {}".format(NUM_IN_OVERFIT,LR_FACTOR,NUM_WARMUP,DROPOUT,BETA,NHEADS,NLAYERS))

print("Evaluating overfit on ALL {} train instances".format(len(trainloader.dataset)))

no_overfit_correct = 0
no_overfit_false = 0
no_mean_correct = 0
no_mean_false = 0
no_med_correct = 0
no_med_false = 0
train_save_struct = []

with torch.no_grad():
    for i, data in enumerate(trainloader):
        signal = data["signal"]
        dSignal = get_deep_seq(data)
        target = data["target_centered"]
        mask = data["mask"]
        size = data["size"]
        name = data["file"]
        output = model(dSignal,mask)
        batchloss = loss_fn(target,output) #L1 LOSS. Mode: "mean"
        # batchloss = ops.generalized_box_iou_loss(output.to(dtype=torch.float32),target.to(dtype=torch.float32))
        
        accScores = vm.pascalACC(output,target)
        no_overfit_correct += accScores[0]
        no_overfit_false += accScores[1]
        IOU = accScores[2] #for batches is a list
        IOU_tr_li += IOU #concat lists 
        for i in range(len(IOU)):
            if(IOU[i]>0.5):
                train_save_struct.append([name,str(1),IOU[i],target,output,size]) #filename, pred-status: correct(1):false(0), IOU-value, ground-truth, prediction-value 
            else:
                train_save_struct.append([name,str(0),IOU[i],target,output,size])
        print("Filename: {}\n Target: {}\n Prediction: {}\n Loss: {}\n IOU: {}".format(data["file"],data["target"],output,batchloss,IOU))
        trainsettestLosses.append(batchloss)
        
        #fix meanModel to have as many entrys as target-tensor: 
        n_in_batch = target.shape[0]
        meanModel_tmp = meanModel.repeat(n_in_batch,1) #make n_in_batch copies along batch-dimension
        medianModel_tmp = medianModel.repeat(n_in_batch,1)
        
        accScores = vm.pascalACC(meanModel_tmp,target)
        no_mean_correct += accScores[0]
        no_mean_false += accScores[1]
        
        accScores = vm.pascalACC(medianModel_tmp,target)
        no_med_correct += accScores[0]
        no_med_false += accScores[1]
        
    print("---------------------------EVAL on ALL {} overfit-train-images---------------------------".format(len(trainloader.dataset)))    
    print("\nTransformer accuracy with PASCAL-criterium on overfit set: {}/{}, percentage: {}".format(no_overfit_correct,no_overfit_false+no_overfit_correct,no_overfit_correct/(no_overfit_false+no_overfit_correct)))    
    print("\nMean model accuracy with PASCAL-criterium on overfit set: {}/{}, percentage: {}".format(no_mean_correct,no_mean_false+no_mean_correct,no_mean_correct/(no_mean_false+no_mean_correct)))
    print("\nMedian model accuracy with PASCAL-criterium on overfit set: {}/{}, percentage: {}".format(no_med_correct,no_med_false+no_med_correct,no_med_correct/(no_med_false+no_med_correct)))
    print("\nMean IOU is {}".format(sum(IOU_tr_li)/len(IOU_tr_li)))
    if(EVAL==0):
        torch.save(train_save_struct,paramsString+classString+"_"+"test_on_train_results.pth")
        print("\n   Results saved to file: ",paramsString+classString+"/"+classString+"_"+"test_on_train_results.pth")
    else: 
        torch.save(train_save_struct,paramsString+"eval/"+classString+"_"+"test_on_train_results.pth")
        print("\n   Results saved to file: ",paramsString+"eval/"+classString+"/"+classString+"_"+"test_on_train_results.pth")
        
    

# #2. TEST-LOOP ON TEST-SET
# no_test_correct = 0 
# no_test_false = 0
# no_test_mean_correct = 0 
# no_test_mean_false = 0
# no_test_median_correct = 0
# no_test_median_false = 0
# testlosses = []
# correct_false_list = []
# IOU_te_li = []

# # meanModel = get_mean_model(oTrainLoader)
# # medianModel = get_median_model(oTrainLoader)

# model.eval()
# with torch.no_grad():
#     running_loss = 0 
#     for i, data in enumerate(testloader):
#         signal = data["signal"]
#         dSignal = get_deep_seq(data)
#         target = data["target"]
#         mask = data["mask"]
#         name = data["file"]
#         size = data["size"]
#         output = model(dSignal,mask)
#         batchloss = loss_fn(target,output) #L1 Loss
#         #batchloss = ops.generalized_box_iou_loss(output.to(dtype=torch.float32),target.to(dtype=torch.float32))
#         running_loss += batchloss.item()
#         testlosses.append(batchloss.item())
#         accScores = vm.pascalACC(output,target)
#         IOU = accScores[2] #for batches is a list
#         IOU_te_li += IOU #list concatenation
#         for i in range(len(IOU)): 
#             if(IOU[i]>0.5):
#                 correct_false_list.append([name,str(1),IOU[i],target,output,size]) #filename, pred-status: correct(1):false(0), IOU-value, ground-truth, prediction-value 
#             else:
#                 correct_false_list.append([name,str(0),IOU[i],target,output,size])
            
        
#         no_test_correct += accScores[0]        
#         no_test_false += accScores[1]
        
#         n_in_batch = target.shape[0]
#         meanModel_tmp = meanModel.repeat(n_in_batch,1) #make n_in_batch copies along batch-dimension
#         medianModel_tmp = medianModel.repeat(n_in_batch,1)
#         accScores = vm.pascalACC(meanModel_tmp,target)
        
#         no_test_mean_correct += accScores[0]
#         no_test_mean_false += accScores[1]
        
#         accScores = vm.pascalACC(medianModel_tmp,target)
#         no_test_median_correct += accScores[0]
#         no_test_median_false += accScores[1]
        
#         if i!=0 and i%100==0:
#             #note that running_loss is not used for anything else than printing.
#             print("L1-loss on every over batch {}:{}: {}\n".format(i-100,i,running_loss/100))
#             running_loss = 0 
            
#         if i==201:
#             break
        
            


# testmeanAcc = no_test_mean_correct/(no_test_mean_false+no_test_mean_correct)
# testmedianAcc = no_test_median_correct/(no_test_median_false+no_test_median_correct)

# print("---------------EVAL on ALL {} test-images---------------".format(len(testloader.dataset)))
# print("\nTransformer accuracy with PASCAL-criterium: {}/{}, percentage: {}".format(no_test_correct,no_test_false+no_test_correct,no_test_correct/(no_test_false+no_test_correct)))    
# print("\nMean model accuracy with PASCAL-criterium: {}/{}, percentage: {}".format(no_test_mean_correct,no_test_mean_false+no_test_mean_correct,testmeanAcc))    
# print("\nMedian model accuracy with PASCAL-criterium: {}/{}, percentage: {}".format(no_test_median_correct,no_test_median_false+no_test_median_correct,testmedianAcc))    
# print("\nMean IOU is {}".format(sum(IOU_te_li)/len(IOU_te_li)))
# if(EVAL==0):
#     torch.save(correct_false_list,paramsString+classString+"_"+"test_on_test_results.pth")
#     print("\n   Results saved to file: ",paramsString+classString+"/"+classString+"_"+"test_on_test_results.pth")
# else:    
#     torch.save(correct_false_list,paramsString+"eval/"+classString+"_"+"test_on_test_results.pth")
#     print("\n   Results saved to file: ",paramsString+"eval/"+classString+"/"+classString+"_"+"test_on_test_results.pth")















# ###-----------------------------------Plots from here------------------------------------------------

# full_embedded_im = output_no_cls.view(1,14,14,768)
# #SINGLE CHANNEL PLOT
# CH_NO = 0
# channel_N = full_embedded_im.squeeze(0)[:,:,CH_NO].detach().cpu().numpy()
# print(channel_N.shape)
# channel_N_T = channel_N.T

# size = (int(data["size"][0,1].item()),int(data["size"][0,0].item())) #sizes are saved as height,width - you need X,Y
# channel_N_im = Image.fromarray(channel_N_T)
# interp_im = np.array(channel_N_im.resize(size,Image.BILINEAR))

# fig,(ax1,ax2) = plt.subplots(1,2)
# ax1.imshow(interp_im)
# fig.suptitle("Usampled deep-feature extraction for channel {}".format(CH_NO))
# ax2.imshow(sampled_back_image)

# def get_first_n_features_plot(df_rep,im_rescaled,NO_CHANNELS):
#     size = im_rescaled.size
#     df_rep_s = df_rep.squeeze(0)
#     fig,axes = plt.subplots(int(math.sqrt(NO_CHANNELS)),int(math.sqrt(NO_CHANNELS)))
#     axes = axes.flatten()
#     #norm = mpl.colors.Normalize(vmin=-1, vmax=1)
#     channel_data = df_rep_s.detach().cpu().numpy()
#     print(channel_data.shape)
#     for i, a in enumerate(axes):
#         print("processing channel {}".format(i))
#         ch_tmp = channel_data[:,:,i]
#         print(ch_tmp.shape)
#         channel_N_im = Image.fromarray(ch_tmp)
#         interp_im_tmp = np.array(channel_N_im.resize(size,Image.BILINEAR))
#         print(interp_im_tmp.size)
#         a.set_xticks([]) #remove ticks
#         a.set_yticks([]) #remove ticks
#         a.imshow(interp_im_tmp)#norm=norm)
#     fig.suptitle("Unsampled deep-feature-extraction for first {} channels".format(NO_CHANNELS))
#     plt.show()
#     return None
        
    
    
# get_first_n_features_plot(full_embedded_im,sampled_back_image,9)

# def get_first_n_features_plot_dual_pane(df_rep,im_rescaled,NO_CHANNELS):
#     size = im_rescaled.size
#     df_rep_s = df_rep.squeeze(0)
#     channel_data = df_rep_s.detach().cpu().numpy()
#     fig = plt.figure(figsize=(14,5.9),constrained_layout=True)
#     subfigs = fig.subfigures(1,2,width_ratios=[1.5, 1.7])
#     ax_L = subfigs[0].subplots(int(math.sqrt(NO_CHANNELS)),int(math.sqrt(NO_CHANNELS)),sharey=True,sharex=True)
#     ax_L = ax_L.flatten()
#     for i, a in enumerate(ax_L):
#         ch_tmp = channel_data[:,:,i]
#         channel_N_im = Image.fromarray(ch_tmp)
#         interp_im_tmp = np.array(channel_N_im.resize(size,Image.BILINEAR))
#         a.set_xticks([]) #remove ticks
#         a.set_yticks([]) #remove ticks
#         a.set_title("{}".format(i),fontweight="bold", size=18,y=0.98)
#         a.imshow(interp_im_tmp)#,cmap='plasma')#norm=norm)
#     ax_R = subfigs[1].subplots(1,1)
#     ax_R.imshow(im_rescaled)
#     ax_R.set_xticks([])
#     ax_R.set_yticks([])
    
#     #fig.suptitle("Unsampled deep-feature-extraction for first {} channels".format(NO_CHANNELS))
    
#     plt.show()
#     #plt.savefig("deep_feature_representation2.pdf")
#     return None

# def plot_mean_latent_space(df_rep,im_rescaled):
#     out = torch.mean(df_rep.squeeze(0),-1)
#     im = Image.fromarray(out.detach().cpu().numpy())
#     #fig = plt.figure(figsize=(14,8))
#     org_size = (500,375)
#     fig, (ax1,ax2) = plt.subplots(1,2,constrained_layout=True,figsize=(14,8))
#     interp_im = np.array(im.resize(org_size,Image.BILINEAR))
#     ax1.imshow(interp_im)
#     ax1.axis('off')
#     ax2.imshow(im_rescaled)
#     ax2.axis('off')
#     #plt.savefig("mean_feature_map.pdf")
    
#     return None
    

# get_first_n_features_plot_dual_pane(full_embedded_im,sampled_back_image,9)
# plot_mean_latent_space(full_embedded_im,sampled_back_image)

# """