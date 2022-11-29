import sys
import os 
import math 

#exp for checking overfitting on full set


L = [375] #[300] #[30,60,120,200]#[2,4,8,16,32]
EPOCHS = [52,134,46]

num_layers = [1,3,5]
num_heads = 1
EVAL = 1

#length = L[classOC]
for epoch in EPOCHS:
    for NL in num_layers:
    	command = "python3 timm_pipe_no_overfit.py " + str(999) + " "+ str(370) + " " + str(NL) + " " + str(num_heads) + " " + str(epoch) + " " + str(EVAL)
    	#print(command)
    	os.system(command)
