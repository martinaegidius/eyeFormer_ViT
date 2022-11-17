#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:44:57 2022

https://pytorch.org/tutorials/beginner/transformer_tutorial.html

@author: max
"""

import torch 
import math 
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import os

DEBUG = False

PATH = os.path.dirname((__file__))

class TransformerModel(nn.Module):
    def __init__(self,ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model,dropout)
        #define encoding layer-style
        encoder_layers = TransformerEncoderLayer(d_model,nhead,d_hid,dropout)
        #generate encoder-stack from layer-style nlayers times
        self.transformer_encoder = TransformerEncoder(encoder_layers,nlayers)
        #init input for encoder-stack as embedded inputs
        self.encoder = nn.Embedding(ntoken,d_model)
        self.d_model = d_model 
        #a bit cumbersome, the following line
        self.decoder = nn.Linear(d_model,ntoken)
        self.init_weights()
        
    def init_weights(self) -> None: #returns none
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange,initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange,initrange)
        
    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:  #only necessary if no autograd
        """
            src: Tensor of shape [seq_len, batch_size]
            src_mask: Tensor of shape [seq_len,seq_len]
            
            returns: Tensor, [seq_len,batch_size,ntoken]
            
        """
        
        src = self.encoder(src) * math.sqrt(self.d_model) #scaling applied
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,src_mask)
        output = self.decoder(output)
        return output
    
def generate_square_subsequent_mask(sz: int) -> Tensor: 
    "Upper triangular of -inf with zeros on diagonal"
    return torch.triu(torch.ones(sz,sz) * float('-inf'),diagonal=1)

class PositionalEncoding(nn.Module): 
    def __init__(self,d_model:int,dropout:float=0.1,max_len: int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1) #make [max_len,1]-tensor
        print(position)
        print("pos shape: ",position.shape)
        div_term = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model)) #vector of size d_model//2
        print("div_term, shape: ",div_term.shape)
        pe = torch.zeros(max_len,1,d_model) #make [max_len,1,d_modl]-tensor
        pe[:,0,0::2] = torch.sin(position*div_term) #fill from 0 with stepsize 2 with sine of product
        pe[:,0,1::2] = torch.cos(position*div_term) #fill from 1 with stepsize 2 with cos of product
        self.register_buffer('pe',pe) #save to model when saving, but do NOT regard as model.parameters() ie no updates in backprop
        
    def forward(self, x: Tensor) -> Tensor: 
        "Args: tensor, shape [seq_len,batch_size,embedding_dimension]"
        x = x + self.pe[:x.size(0)] #add slice of pe corresponding to input size. Remember: PE is constant independently of input x
        return self.dropout(x)
    
    
#load data 
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator 

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer,train_iter),specials=['<unk>']) #map: map function (tokenizer) to iterable, train_iter
vocab.set_default_index(vocab['<unk>']) #index which is returned when out-of-vocabulary token is queried

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor: 
    "convert raw text into a flattened vector"
    data = [torch.tensor(vocab(tokenizer(item)),dtype=torch.long) for item in raw_text_iter] #get every item in raw_text_iter and apply tokenization, followed by vocab. Save result as list of tensors 
    return torch.cat(tuple(filter(lambda t: t.numel()>0,data))) #for elements with length > 0 (ie valid text), concatenate along vertical dimension, tuple for every input element


#need to create iterators again, as train_iter was eaten by building vocab
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batchify(data: Tensor, bsz: int) -> Tensor: 
    """
    Args: 
        data: Tensor, shape [N]
        bsz: int, batch size
    """
    seq_len = data.size(0) // bsz #figure out how long each batch-sequence should be 
    data = data[:seq_len * bsz] #get all elements which fit in batch. Discard rest
    data = data.view(bsz,seq_len).t().contiguous() #transpose to [bsz,seq_len] and force into same part of memory
    return data


batch_size = 20 
eval_batch_size = 10 
train_data = batchify(train_data,batch_size) #becomes: [seq_len, batch size]
val_data = batchify(val_data,eval_batch_size)
test_data = batchify(test_data,eval_batch_size)

""" no of lines / split 
train: 36718
valid: 3760
test: 4358
"""

#we need paired input-target-sequences for the transformer. Often, in language modelling, the target will be the next line (ie what comes next)

bptt = 35 #subdivide data into chunks of length bptt. 
def get_batch(source: Tensor, i: int) -> tuple[Tensor, Tensor]: 

    """
    Args: 
        Source: tensor with shape [full_seq_len, batch_size]
        i = int
        
    Returns: 
        tuple (data, target). 
            Data has shape [seq_len,batch_size], 
            target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt,len(source) - 1 - i) #get hardest constraint
    data = source[i:i+seq_len] #get next slice 
    target = source[i+1:i+1+seq_len].reshape(-1)#get next offset corresponding to target slice
    #if this seems to weird, look at webpage figure 
    
    return data, target


ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

import copy 
import time 

lossF = nn.CrossEntropyLoss()
LR = 5.0
optimizer = torch.optim.SGD(model.parameters(),lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1.0,gamma=0.95) #multiply LR på gamma every step_size epochs

def train(model: nn.Module) -> None:
    model.train() #enable train-setting
    total_loss = 0.0
    log_interval = 200 
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device) #bptt is length of each chunk
    
    num_batches = len(train_data) // bptt 
    for batch, i in enumerate(range(0,train_data.size(0) - 1,bptt)): #batch becomes loop-counter, i becomes start-index of next batch
        data, targets = get_batch(train_data,i)
        seq_len = data.size(0) #rounded paranthesis bc tuple of tensor 
        if seq_len != bptt: 
            src_mask = src_mask[:seq_len,:seq_len] #delete last entries of mask if last batch is NOT full
        output = model(data,src_mask)
        if DEBUG==True:
            print("Output shape becomes ",output.shape)
            print("Output.view(-1,ntokens).shape becomes: ",output.view(-1,ntokens).shape)
        loss = lossF(output.view(-1,ntokens),targets) #get loss when comparing  
        
        loss.backward() #magic line :-) 
        #prevent gradient-explosion. Modifies grads inplace
        torch.nn.utils.clip_grad_norm(model.parameters(),0.5) #constraint on max size of grad. Norm calculated as of ALL norms in a single vec 
        optimizer.step() #apply grads
        
        optimizer.zero_grad() #RESET grads from earlier batch 
        
        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0: #next epoch - print info
            LR = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000/log_interval 
            cur_loss = total_loss/log_interval 
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {LR:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
        

def evaluate(model: nn.Module, eval_data: Tensor) -> float: 
    model.eval()
    total_loss = 0.0
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad(): 
        for i in range(0, eval_data.size(0)-1,bptt):
            data, targets = get_batch(eval_data,i)
            seq_len = data.size(0)
            if seq_len != bptt: 
                src_mask = src_mask[:seq_len,:seq_len]
            output = model(data,src_mask)
            output_flat = output.view(-1,ntokens)
            total_loss += seq_len * lossF(output_flat, targets).item() #why multiply by seq_len?
            
        return total_loss / (len(eval_data)-1)
    
#loop over number of epochs, and save model if val-loss is smallest so far 
best_val_loss = float('inf') #just for init
epochs = 3
best_model = None

import matplotlib.pyplot as plt 

lossPoints = []
epochPoints = []

for epoch in range(1,epochs +1):
    epoch_start_time = time.time() 
    train(model)
    val_loss = evaluate(model,val_data)
    lossPoints.append(val_loss)
    epochPoints.append(epoch)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss 
        best_model = copy.deepcopy(model)

    scheduler.step()   

if not os.path.exists(PATH+"/outputFiles/"):
    os.mkdir(PATH+"/outputFiles/")
    print("Made dir for output")

fig = plt.figure(1)
plt.plot(epochPoints,lossPoints)
plt.title("Cross-entropy loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig(str(PATH+"/outputData/loss_graph.png"))
