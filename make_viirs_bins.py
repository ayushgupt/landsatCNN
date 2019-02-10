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
import math
import os

tiffFileName=str(sys.argv[1])
# 'viirsDistrictLevel/India_2012-10-01_2012-10-31_500.tif'
jsonFileName=str(sys.argv[2])
# 'viirsDistrictLevel/Census_2011/2011_Dist.geojson'
inputFolder=tiffFileName[:-4]
if not os.path.exists(inputFolder):
    os.makedirs(inputFolder)
csv_outfile_name = inputFolder+".csv" 


countryData = json.loads(open(jsonFileName).read())
for currDistrictFeature in countryData["features"]:
    # currDistrictFeature=countryData["features"][0]
    distName=currDistrictFeature["properties"]['DISTRICT']
    st_cen_cd=currDistrictFeature["properties"]['ST_CEN_CD']
    censuscode=currDistrictFeature["properties"]['censuscode']
    geoms=currDistrictFeature["geometry"]
    listGeom=[]
    listGeom.append(geoms)
    geoms=listGeom
    with rasterio.open(tiffFileName) as src:
      out_image, out_transform = mask(src, geoms, crop=True)

    out_meta = src.meta.copy()

        # save the resulting raster  
    out_meta.update({"driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform})

    with rasterio.open(inputFolder+'/'+distName+'@'+str(st_cen_cd)+'@'+str(censuscode)+".tif", "w", **out_meta) as dest:
      dest.write(out_image)



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


printing_dictionary={}
for key, val in flattened_DataDictionary.items():
    tempArray=val.copy()
    tempArray[tempArray>10]=10
    tempArray=tempArray+1.0
    bins_1 = np.array([t/10 for t in range(102)])
    binning=np.histogram(tempArray, bins=bins_1)
    str1=key
    str2=str1[:-4]
    distName_st_cen_cd_censuscode=str2.split('@')
    currArray=np.array([int(distName_st_cen_cd_censuscode[1]),int(distName_st_cen_cd_censuscode[2])])
    currArray=np.append(currArray,binning[0])
    printing_dictionary[distName_st_cen_cd_censuscode[0]]=currArray



columns1=['st_cen_cd','censuscode']
col_help=['light_'+str(t) for t in range(101)]
columns1.extend(col_help)
dataframe_districts=pd.DataFrame.from_dict(printing_dictionary, orient='index',columns=columns1)
dataframe_districts.to_csv(csv_outfile_name)