#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 12:03:42 2022

@author: max
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math

class eyeFormer(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout=0):
        self.d_model = input_dim
        super(eyeFormer,self).__init__() #get class instance
        self.pos_encoder = PositionalEncoding(input_dim,dropout)
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=input_dim,nhead=2,dim_feedforward=hidden_dim,dropout=0,activation="relu",batch_first=True)
        #make encoder - 3 pieces
        self.encoder = nn.TransformerEncoder(self.encoderLayer,num_layers = 3)
        #make decoder 
        self.fc = nn.Linear(input_dim,output_dim,bias=True)
            
    def forward(self,x):
        x = x* math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.encoder(x)
        output = self.fc(output)
        return output
    
    

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout = 0.0,max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len,1,d_model)
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
    

model = eyeFormer(2,2048,4)

class make_data(Dataset):
    def __init__(self):
        self.x = torch.rand((1000,32,2))
        self.y = torch.rand((1000,4))
        self.len = 1000
        
    def __getitem__(self,index):
        return self.x[index],self.y[index]
        
    def __len__(self):
        return self.len
    
traindata = make_data()
testdata = make_data()
    
train_dataloader = DataLoader(dataset=traindata,batch_size=4,shuffle=True)
test_dataloader = DataLoader(dataset=testdata,batch_size=4,shuffle=True)

for batch, (x,y) in enumerate(train_dataloader):
   print(f"Batch: {batch+1}")
   print(f"X shape: {x.shape}")
   print(f"y shape: {y.shape}")
   break



learning_rate = 0.1
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

num_epochs = 2
loss_values = []
for epoch in range(num_epochs):
    for x, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()
       
        # forward + backward + optimize
        pred = model(x)
        loss = loss_fn(pred, y.type(torch.LongTensor))
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

print("Training Complete")

