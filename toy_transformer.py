#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:44:36 2022

Implementation of: Attention is all you need Transformer model

"""
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad 
import copy #for deepcopy
import math
import warnings


# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

#HELPER_FUNCTIONS

def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None



class EncoderDecoder(nn.Module):
    #define a standard encoder-decoder architecture. For now looks quite abstract 
    
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super().__init__() #call parent or sibling class, in this case nn.Module
        self.encoder = encoder 
        self.decoder = decoder 
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator 
        
    def forward(self,src,tgt,src_mask,tgt_mask):
        #Take input and process with mask and target sequences
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)
    
    def encode(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)
    
    
    
class Generator(nn.Module):
    #defines a standard linear + softmax generation step, ie last part of transformer, which projects from model-dimensions to vocabulary, and softmaxes for probabilities.
    def __init__(self,d_model,vocab):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab) #d_model is input of linear layer, and simply is dimensionality of model output. vocab is size of vocabulary + <eof>
        
    def forward(self,x):
        return log_softmax(self.proj(x),dim=-1) #can not explain why -1. 


def clones(module,N): #produce N identical layers
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)]) #list-comprehension. Syntax: make a list of length N with deepcopy of Module-item and transform to nn

class Encoder(nn.Module):
    #build a core encoder as a stack of N layers
    
    def __init__(self,layer,N):
        super().__init__() #parent: nn.Module
        self.layers = clones(layer,N) #create stack
        self.norm = LayerNorm(layer.size) #define output-function
        #Above line throws error
        
    def forward(self,x,mask):
        #pass input and mask through each layer in turn
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class LayerNorm(nn.Module):
    #Create layernorm module. We want a residual connection around each sub-layer, followed by a layer normalization ("Add & Normalize"). 
    #Layer-normalization is implemented here, residual connection is created by saying: output = LayerNorm(x+Sublayer(x)).
    
    def __init__(self,features,eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features)) #function creates a tensor which is considered a module parameter, and thus can be optimized
        self.b_2 = nn.Parameter(torch.zeros(features)) #b_2: bias-term. a_2: weight-term.
        self.eps = eps 
        
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True) #dim = -1 means along LAST dimension (backward indexing)
        std = x.std(dim=-1,keepdim = True)
        return self.a_2*(x-mean)/(std+self.eps) + self.b_2
        #we add + eps for preventing zeros 
        

        
        
class SublayerConnection(nn.Module):
    #Create Layernorm(x+ Sublayer(x))
    def __init__(self,size,dropout):
        super(SublayerConnection,self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,sublayer): #give sublayer a residual connection 
        return x + self.dropout(sublayer(self.norm)) #sublayer is "inner function" of layer
    
    
    
#now, we may finally implement encoding-layer, which consists of 2 sublayers; 1. selfattention(multihead), 2. fully-connected (feed forward)
class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size,dropout),2) #make two sublayer-instances of yet undefined type 
        self.size = size
        
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,mask)) #magical line, do not understand. Shoot through layer type 1 for all inputs 
        return self.sublayer[1](x,self.feed_forward) #shoot through layer 2 and return 
    
#Decoder is the same, but reversed: 
class Decoder(nn.Module):
    #with masking 
    def __init__(self,layer,N):
        super().__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size) #Layer size = number of features 
    
    def forward(self, x, memory, src_mask,tgt_mask):
        for layer in self.layers: 
            x = layer(x,memory,src_mask,tgt_mask) 
        return self.norm(x)
        
class DecoderLayer(nn.Module):
    #Decoder-block is built of self-attn, src attn and feed forward
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super().__init__()
        self.size = size 
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size,dropout),3)

    def forward(self,x,memory,src_mask,tgt_mask):
        m = memory 
        x = self.sublayer[0](x,lambda x: self.self_attn(x,x,x,tgt_mask)) #self-attention        
        x = self.sublayer[1](x,lambda x: self.srcattn(x,m,m,src_mask))
        return self.sublayer[2](x,self.feed_forward)
    
def subsequent_mask(size):
    ##mask out later positions. Function for creating a matrix of rows indicating word under analysis and column meaning words in sequence which may be inspected during analysis. 
    attn_shape = (1,size,size)
    subsequent_mask = torch.triu(torch.ones(attn_shape),diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def attention(query,key,value,mask=None,dropout=None):
    #get scaled dot product attention
    #outputs attention and softmax of initial score 
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)  #transpose: simply switches last two dimensions!
    if mask is not None: 
        scores = scores.masked_fill(mask==0,-1e9)
    p_attn = scores.softmax(dim=-1)#softmax, last dimension
    if dropout is not None: 
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value), p_attn
    
    
#¤now we may build multihead-attention

class MultiHeadedAttention(nn.Module): #for eight parallel attention heads
    def __init__(self,h,d_model,dropout = 0.1):
        super().__init__()
        assert d_model % h == 0 #ensure that h scales correctly with d_model ie is a power-of-two-fraction
        self.d_k = d_model // h #int division
        self.h = h 
        self.linears = clones(nn.Linear(d_model,d_model),4) #FOUR COPIES - WHY NOT 8? 
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self,query,key,value,mask=None):
        if mask is not None: 
            mask = mask.unsqueeze(1) #for making it (1,1,size,size). Same mask is used for all heads
        nbatches = query.size(0) #number of seperate queries 
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x) #for some reason last layer. Unsure why. I believe we have a depth of 4, and thus only want last layer
        
#Now we must build a fc feed-forward network on top of all heads

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model,d_ff) #dff is 4*d_model
        self.w_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
        
        def forward(self,x):
            return self.w_2(self.dropout(self.w_1(x).relu()))
        
        
        #Note: they are the same across different positions of input, but are different between layers in decoder. 
        #ie one fc pr. decoder-block
        
        
class Embeddings(nn.Module):
    #remember that embeddings are LEARNED. 
    def __init__(self, d_model, vocab): 
        super().__init__()
        self.lut = nn.Embedding(vocab,d_model) #nn.Embeddings creates a look-up table by using size of vocabulary and d_model (size of each embedding vector). Output is matrix (*,H) with *: num of inputs and H: size of each embedding vec, ie d_model
        self.d_model = d_model
        
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model) #lut still takes input, as lut is a FUNCTION 

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def inference_test():
    test_model = make_model(11,11,2)
    test_model.eval()
    src = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]])
    src_mask = torch.ones(1,1,10)
    
    memory = test_model.encode(src,src_mask)
    # ys = torch.zeros(1,1).type_as(src)
    
    # for i in range(9):
    #   out = test_model.decode(
    #       memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
    #   )
    #   prob = test_model.generator(out[:, -1])
    #   _, next_word = torch.max(prob, dim=1)
    #   next_word = next_word.data[0]
    #   ys = torch.cat(
    #       [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
    #   )

    # print("Example Untrained Model Prediction:", ys)
    
def run_tests():
    for _ in range(10):
        inference_test()
        
show_example(run_tests)
    
        
        