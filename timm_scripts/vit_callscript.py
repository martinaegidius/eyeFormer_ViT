import sys
import os 
import math 

#exp for running 20% val to get the overfitting-point 

classOC = int(sys.argv[1])


L = [4,8,16,32]#[2,4,8,16,32]
EPOCHS = 800

num_layers = 6
num_heads = 2
EVAL = 0

#length = L[classOC]
for length in L:
    command = "python3 timm_pipe_cmd.py " + str(classOC) + " "+ str(length) + " " + str(num_layers) + " " + str(num_heads) + " " + str(EPOCHS) + " " + str(EVAL)
    #print(command)
    os.system(command)
