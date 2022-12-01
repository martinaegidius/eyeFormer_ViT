import sys
import os 
import math 

#exp for checking overfitting on full set


L = [999] #[300] #[30,60,120,200]#[2,4,8,16,32]
EPOCHS = [120,120,120]

num_layers = [5,6,9]
num_heads = 1
EVAL = 1

#length = L[classOC]
for i, epoch in enumerate(EPOCHS):
    command = "python3 timm_pipe_no_overfit.py " + str(999) + " "+ str(999) + " " + str(num_layers[i]) + " " + str(num_heads) + " " + str(epoch) + " " + str(EVAL)
    #print(command)
    os.system(command)
