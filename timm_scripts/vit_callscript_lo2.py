import sys
import os 
import math 

#exp for checking overfitting on full set


L = [375] #[300] #[30,60,120,200]#[2,4,8,16,32]
EPOCHS = 350

num_layers = [1,3]
num_heads = 1
EVAL = 0

#length = L[classOC]
for length in L:
    for NL in num_layers:
    	command = "python3 timm_pipe_cmd.py " + str(999) + " "+ str(length) + " " + str(NL) + " " + str(num_heads) + " " + str(EPOCHS) + " " + str(EVAL)
    	#print(command)
    	os.system(command)
