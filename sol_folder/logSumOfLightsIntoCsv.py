import math
import json
import sys
import pandas as pd
from libtiff import TIFF
import sys
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import pickle
import h5py
import scipy.misc
import seaborn as sns
import math
import os
import matplotlib.pyplot as plt

inputFolder=sys.argv[1]
#inputFolder='sol_folder/states_all_2014-12-01_2014-12-31_30'
if not os.path.exists(inputFolder):
    os.makedirs(inputFolder)
    
    
onlyfiles = [f for f in listdir(inputFolder) if isfile(join(inputFolder, f))]
flattened_DataDictionary={}


for currDFile in onlyfiles:
    currDistrictFile=(inputFolder+'/'+currDFile)
    tif = TIFF.open(currDistrictFile, mode='r')
    image = tif.read_image()
    dataAll = np.array(image)
    flattenData=dataAll.flatten()
    flattenData=flattenData[flattenData != 0]
    flattened_DataDictionary[currDFile]=flattenData
        
        
        
file1 = open(inputFolder+'_sol.csv',"w")
file1.write("distName,st_cen_cd,censuscode,SOL\n")
for state,vec in flattened_DataDictionary.items():
    list_temp=state[:-4].split('@')
    list_temp.append(str(vec.sum()))
    file1.write(','.join(list_temp)+'\n')