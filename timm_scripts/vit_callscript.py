import sys
import os 
import math 

#exp for checking overfitting on full set


L = [30,60,120,200]#[2,4,8,16,32]
EPOCHS = 500

num_layers = 3
num_heads = 1
EVAL = 0

#length = L[classOC]
for length in L:
    command = "python3 timm_pipe_cmd.py " + str(999) + " "+ str(length) + " " + str(num_layers) + " " + str(num_heads) + " " + str(EPOCHS) + " " + str(EVAL)
    #print(command)
    os.system(command)
