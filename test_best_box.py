#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:02:42 2022

@author: max
"""

import os 


from load_POET import pascalET
import numpy as np
import time
import copy 

DEBUG = False
DEBUG_CLEANUP = False
dset = pascalET()
dset.loadmat()


#debug new format 
dset.convert_eyetracking_data(CLEANUP=True,STATS=True)
dset.load_images_for_class(0)



#test for image with many airplanes
idx = 12
f = dset.etData[0][idx].filename + ".jpg"
tmp_box = dset.etData[0][12].gtbb #image with many planes
#fixArr = dset.debug_box_BP[191]  #whoops - runs through whole dset, and fixArr is del in every loop. Need to recreate fixArr
#quick-fix:
for k in range(len(dset.eyeData[0][12])):
    BP = dset.eyeData[0][12][k][0] #(is a list :/)
    if(k==0): #create fixArr
        if(BP.shape==(2,0)):
            fixArr = copy.deepcopy(np.transpose(BP)) #invalid measurements are for some reason stored as shape (0,2) (transpose of other measurements)
        else:
            fixArr = copy.deepcopy(BP)
    else:
        if(BP.shape[1] == fixArr.shape[1]): #necessary check as None array can not be concat
            fixArr = np.vstack((fixArr,BP)) 
        else: 
            pass

dset.get_bounding_box(tmp_box,fixArr)
print("Number of fixes in general: ",dset.eyeData_stats[0,12,0])
print("Number of fixes in chosen bbox: ", dset.eyeData_stats[0,12,1])
dset.specific_plot(0,f)




"""
dset.specific_plot(0,"2009_004601.jpg")
idx = dset.filename.index("2009_004601.jpg") #remember - this only works with current filename structure. Right now, filenames are saved when images are loaded
#for debugging best bbox: following image has two boxes
testImB = dset.etData[0][17].gtbb
testBP = dset.debug_box_BP[17] 
print("Complete fixArr: ",testBP)

dset.get_bounding_box(testImB,testBP)


numsOfBoxes = []
for i in range(len(dset.etData[0])):
    numsOfBoxes.append(dset.etData[0][i].gtbb.shape)

#final bbox debugging
imsToInvestigate = [1,2,3,12] #for image 1 correct, 2 correct, 3 correct, 12 correct
for i in imsToInvestigate:
    f = dset.etData[0][i].filename + ".jpg"
    idx = dset.filename.index(f)
    testImB = dset.etData[0][i].gtbb
    testBP = dset.debug_box_BP[i]
    dset.get_bounding_box(testImB,testBP)
    dset.specific_plot(0,f)
    time.sleep(3)
    print("Check number of detected fixes in box: ")
    try:
        print("Number of fixes in general: ",dset.eyeData_stats[0,idx,:,0])
        print("Number of fixes in bbox: ", dset.eyeData_stats[0,idx,:,1])
        #seems right, but IT DOES NOT CHECK OTHER BOX!
    except:
        pass
    input("Press Enter to continue...")

print("Check number of detected fixes in box: ")
try:
    print("Number of fixes in general: ",dset.eyeData_stats[0,idx,:,0])
    print("Number of fixes in bbox: ", dset.eyeData_stats[0,idx,:,1])
    #seems right, but IT DOES NOT CHECK OTHER BOX!
except:
    pass

"""