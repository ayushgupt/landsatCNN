import math
import json
import sys
import pandas as pd
import rasterio
from rasterio.mask import mask
# from libtiff import TIFF
# import libtiff
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

originalFolder = sys.argv[1] # 'data/2014_dec'
cutFilesFolder = originalFolder+'_cutFiles'
csvFileName = cutFilesFolder+'.csv'

onlyfiles = [f for f in listdir(cutFilesFolder) if isfile(join(cutFilesFolder, f))]

village_File_Dictionary={}
for tempStr in onlyfiles:
    splitUpList=tempStr[:-4].split('@')
    splitUpList=splitUpList[1:]
    keyVillage = '@'.join(splitUpList)
    if (not (keyVillage in village_File_Dictionary)):
        village_File_Dictionary[keyVillage]=[tempStr]
    else:
        village_File_Dictionary[keyVillage].append(tempStr)
        
        
# 0.2 from -10 to -8
# 0.1 from -8 to 3 
# 0.2 from 3 to 5
range1 = list(range(-100,-80,2))
range2 = list(range(-80,30,1))
range3 = list(range(30,50,2))
(range1.extend(range2))
range1.extend(range3)
rangeBins = [t/10 for t in range1]
# print(rangeBins)

columns1=['vCode2001','vid','vCode2011']
col_help=['light_'+str(t) for t in range(129)]
columns1.extend(col_help)
binningDf = pd.DataFrame(columns=columns1)

village_bins_dict={}
numVillageC = 0
for keyVillage,villageFileList in village_File_Dictionary.items():
    numVillageC = numVillageC + 1
    currVillage_pixels= None
    for cVillageFile in villageFileList:
        #print(cVillageFile)
        im = Image.open(cutFilesFolder+'/'+cVillageFile) 
        imarray = np.array(im)
        flattenData=imarray.flatten()
        flattenData=flattenData[flattenData != 0]
        if(currVillage_pixels is None):
            currVillage_pixels = flattenData
        else:
            currVillage_pixels = np.append(currVillage_pixels,flattenData)
    currVillage_pixels=np.log(currVillage_pixels)
    currVillage_pixels = currVillage_pixels[~np.isnan(currVillage_pixels)]
    currVillage_pixels[currVillage_pixels<(-10.0)] = (-10.0)
    currVillage_pixels[currVillage_pixels>4.8] = (4.8)
    binning=np.histogram(currVillage_pixels, bins=rangeBins)
    list_row=(keyVillage.split('@'))
    list_row.extend(binning[0])
    binningDf.loc[numVillageC-1] =  list_row
    
binningDf.to_csv(csvFileName)