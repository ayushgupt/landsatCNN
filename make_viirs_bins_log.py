import math
import json
import sys
import pandas as pd
import rasterio
from rasterio.mask import mask
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

# tiffFileName='viirsDistrictLevel/India_2012-10-01_2012-10-31_500.tif'
# jsonFileName='viirsDistrictLevel/Census_2011/2011_Dist.geojson'
tiffFileName=str(sys.argv[1])
# 'viirsDistrictLevel/India_2012-10-01_2012-10-31_500.tif'
jsonFileName=str(sys.argv[2])
# 'viirsDistrictLevel/Census_2011/2011_Dist.geojson'
inputFolder=tiffFileName[:-4]
csv_outfile_name1 = inputFolder+"_log.csv" 

onlyfiles = [f for f in listdir(inputFolder) if isfile(join(inputFolder, f))]
flattened_DataDictionary={}

for currDFile in onlyfiles:
    #currDistrictFile='districtTiffFiles/Rajkot@24@476.tif'
    currDistrictFile=(inputFolder+'/'+currDFile)
    tif = TIFF.open(currDistrictFile, mode='r')
    image = tif.read_image()
    dataAll = np.array(image)
    flattenData=dataAll.flatten()
    flattenData=flattenData[flattenData != 0]
    flattened_DataDictionary[currDFile]=flattenData


printing_dictionary_log={}
for key, val in flattened_DataDictionary.items():
    tempArray1=val.copy()
    tempArray2=np.log(tempArray1)
    tempArray = tempArray2[~np.isnan(tempArray2)]
    tempArray[tempArray>3]=3
    tempArray[tempArray<(-4)]=-4
    bins_1 = np.array([t/10 for t in range(-41,32)])
    binning=np.histogram(tempArray, bins=bins_1)
    str1=key
    str2=str1[:-4]
    distName_st_cen_cd_censuscode=str2.split('@')
    currArray=np.array([int(distName_st_cen_cd_censuscode[1]),int(distName_st_cen_cd_censuscode[2])])
    currArray=np.append(currArray,binning[0])
    printing_dictionary_log[distName_st_cen_cd_censuscode[0]]=currArray




columns1=['st_cen_cd','censuscode']
col_help=['light_'+str(t) for t in range(72)]
columns1.extend(col_help)
dataframe_districts=pd.DataFrame.from_dict(printing_dictionary_log, orient='index',columns=columns1)
dataframe_districts.to_csv(csv_outfile_name1)

