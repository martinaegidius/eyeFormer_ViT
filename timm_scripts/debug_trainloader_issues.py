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
import sys
#import handle_pascal as hdl
from load_POET_timm import pascalET
import box_operations as boxops

###CALL WITH SYSTEM ARGUMENTS
#arg 1: classChoice
#arg 2: NUM_IN_OVERFIT
#arg 3: NUM_LAYERS
#arg 4: NUM_HEADS
 
#try:
#    classChoice = int(sys.argv[1])
#except:
#    classChoice = None
#    pass
classChoice = [x for x in range(10)]
#classChoice = [0]

SAVEDSET = False #flag for defining if data-sets are saved to scratch after dataloader generation. For ViT false because big files. 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("FOUND DEVICE: ",device)


def generate_DS(dataFrame,classes=[x for x in range(10)]):
    """
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
    dataFrame.convert_eyetracking_data(CLEANUP=True,STATS=True,mode="mean") #MEAN FIXATION EXPERIMENT 
    NUMCLASSES = len(classes) 
    
    
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

class PASCALdataset(Dataset):
    def __init__(self,pascalobj,root_dir,classes,sTransform=None,imTransform = None,coordTransform = None):
        self.ABDset,self.filenames,self.eyeData,self.targets,self.classlabels,self.imdims,self.length = generate_DS(pascalobj,classes)
        self.root_dir = root_dir
        self.sTransform = sTransform
        self.imTransform = imTransform
        self.coordTransform = coordTransform
        self.deep_representation = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ViT_model = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=0, global_pool='')
        self.ViT_model.eval()
        
    def __len__(self): 
        return len(self.ABDset)
    
    
    def __getitem__(self,idx): #for images
        if torch.is_tensor(idx):
            idx = idx.tolist() #for indexing
        
        SINGLE_PARTICIPANT = True
        
        
        filename = self.filenames[idx]
        image = Image.open(self.root_dir + filename)
        
       
        if(SINGLE_PARTICIPANT==True):
            signals = self.eyeData[idx][0]#participant zero only
        
        else: 
            signals = self.eyeData[idx,:]
        
        classL = self.classlabels[idx]
        
        targets = self.targets[idx,:]
        size = self.imdims[idx]
        
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

    n_train_samples = []
    training_data = []
    for classes in classesOC: #loop saves in list of lists format
        train_filename = split_dir + classlist[classes]+"_Rbbfix.txt"
        with open(train_filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        for i in range(len(lines)): 
            lines[i] = classlist[classes]+"_"+lines[i]+".jpg"
            
        training_data.append(lines)
        n_train_samples.append(len(lines))
        
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
         
    return train,test,n_train_samples

def get_balanced_permutation(nsamples,NUM_IN_OVERFIT,valsize=100):
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
        vidx = t_idx[classwise_nsamples[num]:classwise_nsamples[num]+int(valsize/len(nsamples))]
        ofIDX = torch.cat((ofIDX,idx),0)
        valIDX = torch.cat((valIDX,vidx),0)
        offset += instance
        
    return ofIDX,valIDX,NUM_IN_OVERFIT


def load_split(className,root_dir):
    #loads earlier saved splits from disk. 
    #Arg: className (e.g. "airplane")
    
    tmp_root = root_dir+"/datasets/"
    print("................. Loaded ",className," datasets from disk .................")
    train = torch.load(tmp_root+className+"Train.pt") #load earlier saved split
    test = torch.load(tmp_root+className+"Test.pt")
    return train,test


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]
if(classChoice!=None):
    classesOC = classChoice
else:
    print("No selection of data provided. Used class 3, 2, 1.")
    classesOC = [3,2,1]
    #classChoice = 3

torch.manual_seed(9)
###SCRIPT PARAMETERS###
GENERATE_DATASET = True
timm_root = os.path.dirname(__file__)
if((len(classesOC)>1) and (len(classesOC)<10)):
    classString = "multiple_classes"
elif(len(classesOC)==10):
    classString = "all_classes"
else: 
    classString = classes[classesOC[0]]

BATCH_SZ = 8
# #RUN_FROM_COMMANDLINE. Class at top of programme.
NUM_IN_OVERFIT = 400 #NUM-IN-OVERFIT EQUALS LEN(TRAIN) IF OVERFIT == False
NLAYERS = 1
NHEADS = 1
EPOCHS = 10
EVAL = 0

#NLAYERS = 3
#NHEADS = 1
#EPOCHS = 600
#NUM_IN_OVERFIT = 2
DROPOUT = 0.0
LR_FACTOR = 1
EPOCH_STEPS = NUM_IN_OVERFIT//BATCH_SZ
if EPOCH_STEPS == 0: 
    EPOCH_STEPS = 1
NUM_WARMUP = int(EPOCHS*(1/3))*EPOCH_STEPS #constant 30% warmup-rate
if NUM_WARMUP == 0: #debugging cases
    NUM_WARUP = 1

BETA = 1
OVERFIT=True
#EVAL = 0 #flag which is set for model evaluation, ie. final model. Set 1 if it is final model.

   
        


    
if(GENERATE_DATASET == True or GENERATE_DATASET==None):
    dataFrame = pascalET()
    #root_dir = os.path.dirname(__file__) + "/../eyeFormer/Data/POETdataset/PascalImages/"
    root_dir = os.path.join(os.path.expanduser('~'),"BA/eyeFormer/Data/POETdataset/PascalImages/")
    split_root_dir = os.path.join(os.path.expanduser('~'),"BA/eyeFormer")
    
    SignalTrans = torchvision.transforms.Compose(
        [torchvision.transforms.Lambda(boxops.tensorPad())])
        #,torchvision.transforms.Lambda(rescale_coords())]) #transforms.Lambda(tensor_pad() also a possibility, but is probably unecc.
    
    ImTrans = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224),torchvision.transforms.InterpolationMode.BILINEAR),
                                              torchvision.transforms.ToTensor()]) #should be normalized and converted to 0,1. ToTensor always does this for PIL images
    
    CoordTrans = torchvision.transforms.Compose([torchvision.transforms.Lambda(boxops.rescale_coords()),torchvision.transforms.Lambda(boxops.get_center())])
    

    fullDataset = PASCALdataset(dataFrame, root_dir, classesOC,sTransform=SignalTrans,imTransform = ImTrans,coordTransform = CoordTrans) #init dataset as torch.Dataset.
    
    trainLi,testLi,nsamples = get_split(split_root_dir,classesOC) #get Dimitrios' list of train/test-split.
    trainIDX = [fullDataset.filenames.index(i) for i in trainLi] #get indices of corresponding entries in data-structure
    testIDX = [fullDataset.filenames.index(i) for i in testLi]
    #subsample based on IDX
    train = torch.utils.data.Subset(fullDataset,trainIDX)
    test = torch.utils.data.Subset(fullDataset,testIDX)
    #save generated splits to scratch
    if(SAVEDSET == True):
        torch.save(train,timm_root+"/datasets/"+classString+"Train.pt")
        torch.save(test,timm_root+"/datasets/"+classString+"Test.pt")
        print("................. Wrote datasets for ",classString,"to disk ....................")
        

if(GENERATE_DATASET == False):
    train,test = load_split(classString,timm_root)  
    print("Succesfully read data from binary")
 

#make dataloaders of chosen split
trainloader = DataLoader(train,batch_size=BATCH_SZ,shuffle=True,num_workers=1)
#FOR DEBUGGING
#for i, data in enumerate(trainloader):
#    if i==1:
#        DEBUGSAMPLE = data

testloader = DataLoader(test,batch_size=BATCH_SZ,shuffle=True,num_workers=1)

if NUM_IN_OVERFIT==None and OVERFIT==True: #only if no cmd-argv provided
    NUM_IN_OVERFIT=16
    print("...No argument for length of DSET provided. Used N=16...")
    
g = torch.Generator()
g.manual_seed(1)

if(OVERFIT): #CREATES TRAIN AND VALIDATION-SPLIT 
     #new mode for getting representative subsample: 
    ofIDX,valIDX,NUM_IN_OVERFIT = get_balanced_permutation(nsamples,NUM_IN_OVERFIT,valsize=60)
    #print(valIDX)
    #print("valIDX shape: ",valIDX.shape)
    
    #IDX = torch.randperm(len(train),generator=g[0])#[:NUM_IN_OVERFIT].unsqueeze(1) #random permutation, followed by sampling and unsqueezing
    #ofIDX = IDX[:NUM_IN_OVERFIT].unsqueeze(1)
    #valIDX = IDX[NUM_IN_OVERFIT:NUM_IN_OVERFIT+17].unsqueeze(1) #17 because the smallest train-set has length 33=max(L)+17.
    overfitSet = torch.utils.data.Subset(train,ofIDX)
    valSet = torch.utils.data.Subset(train,valIDX)
    trainloader = DataLoader(overfitSet,batch_size=BATCH_SZ,shuffle=True,num_workers=0,generator=g)
    print("Overwrote trainloader with overfit-set of length {}".format(ofIDX.shape[0]))
    valloader = DataLoader(valSet,batch_size=BATCH_SZ,shuffle=True,num_workers=0,generator=g)
    print("Valloader of constant length {}".format(valIDX.shape[0]))

    
    
    #old implementation (single-class)
    #IDX = torch.randperm(len(train),generator=g)#[:NUM_IN_OVERFIT].unsqueeze(1) #random permutation, followed by sampling and unsqueezing
    #ofIDX = IDX[:NUM_IN_OVERFIT].unsqueeze(1)
    #valIDX = IDX[NUM_IN_OVERFIT:NUM_IN_OVERFIT+17].unsqueeze(1) #17 because the smallest train-set has length 33=max(L)+17.
    #overfitSet = torch.utils.data.Subset(train,ofIDX)
    #valSet = torch.utils.data.Subset(train,valIDX)
    #trainloader = DataLoader(overfitSet,batch_size=BATCH_SZ,shuffle=True,num_workers=1,generator=g)
    #print("Overwrote trainloader with overfit-set of length {}".format(ofIDX.shape[0]))
    #valloader = DataLoader(valSet,batch_size=BATCH_SZ,shuffle=True,num_workers=1,generator=g)
    #print("Valloader of constant length {}".format(valIDX.shape[0]))


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
    
     
# output_no_cls = output[0,1:,:]
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
    signal = sample["signal"].to(device)
    if signal.ndim==2: #fix single batches which come without batch. Should not be an issue using trainloader, only when debugging
        signal = signal.unsqueeze(0)
        
    BSZ = signal.shape[0]
    featuremap = sample["image"].to(device)
    size = sample["size"].to(device)
    CLS = featuremap[:,:,0,:] #pull out cls-tokens. Gets all batches, size [batch_sz,1,1,768]
    featuremap_noCLS = featuremap[:,:,1:,:]
    featuremap_noCLS = featuremap_noCLS.view(BSZ,14,14,768)
    featuremap_perm = featuremap_noCLS.permute(0,-1,1,2) #bilinear interp runs over last two dimensions. Therefore permute to [batch_sz,768,x,y]
    holder_t = torch.zeros(BSZ,signal.shape[1],featuremap.shape[-1]+2).to(device)
    #holder_t = torch.zeros(BSZ,signal.shape[1],770)
    featuremap_holder_l = []
    for i in range(BSZ):
        sizes = (int(size[i][1].item()),int(size[i][0].item())) #x,y. Remember data["batch"] is h,w
        if(sizes!=(0,0)):
            #interpolate function fails for uninitialized lenghts 
            featuremap_intpl = F.interpolate(featuremap_perm[i].unsqueeze(0),sizes,mode='bilinear') #interpolate tensor. Sadly cant be done outside of loop, as sizes to interpolate to have to be constant (cant be tensor)
            featuremap_intpl = featuremap_intpl.permute(0,2,3,1) #get back to format [batch_sz,x,y,embed_dim]
            featuremap_holder_l.append(featuremap_intpl)
            #now get indices by slicing 
            x = signal[i,:,0].to(dtype=torch.long)
            y = signal[i,:,1].to(dtype=torch.long)
            xslice = x
            yslice = y
            holder_slice = featuremap_intpl[(slice(None),xslice,yslice,slice(None))] #returns sliced featuremap. Format: [1,(seq_len),768]
        #   print(holder_t.shape)
            holder_t[i,:,2:] = holder_slice
        #print("Shape after loop",holder_t.shape)    
        #concat CLS and normalized x,y
        #normalize x and y to be between 0 and 1. Worked fine with raw signal - but now we use signal normalized relative to image-size instead (still between 0,1 but relative to image)
            #norm_sig = F.normalize(signal,p=2,dim=1) #each feature column is normalized per entry to fit in unit-interval. 
            #holder_t[:,:,0] = norm_sig[:,:,0] #fill in x-coords
            #holder_t[:,:,1] = norm_sig[:,:,1] #fill in y-coords aswell. Resultant format: channel 0: y, channel 1: x,[batch,y,x] 
            #print("Shape after app. xy",holder_t.shape)
        else: 
            print("Discovered empty shape-array for sample {} i in batch!. Filename: {}, index {}".format(i,sample["file"][i],sample["index"]))
            print(sizes)
            featuremap_holder_l.append(None)
            break

    holder_t[:,:,0] = sample["scaled_signal"][:,:,0]
    holder_t[:,:,1] = sample["scaled_signal"][:,:,1]
        
    if(get_feature_map==True):
        return holder_t,featuremap_holder_l
    else:
        return holder_t

print("testloader len is: ",len(testloader))
# for i, data in enumerate(tqdm(testloader)):
#     zeros = (data["size"]==0).sum() #calculate zeros across all dims
#     zeros_li = []
#     if zeros != 0:
#         print("Detected zero-size in batch {} with filenames {}".format(i,data["file"]))
#         print(data["size"])
#     zeros_li.append(zeros)

# print(zeros_li)

def debug_get_deep_seq(testloader):
    for j, data in enumerate(tqdm(testloader)):
        BS = data["size"].shape[0]
        size = data["size"]
        for i in range(BS):
            sizes = (int(size[i][1].item()),int(size[i][0].item()))
            zeros = sizes.count(0)
            zerosF = sizes.count(0.)
            if zeros!=0 or zerosF != 0: 
                print("Found issue in batch {}, sample {}, filename ".format(j,i,data["file"][i]))
                print("Sizes: {}".format(data["size"][j]))
        dSignal = get_deep_seq(data).to(device)

debug_get_deep_seq(testloader)
