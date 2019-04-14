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

#tiffFileName='viirsDistrictLevel/India_2012-10-01_2012-10-31_500.tif'
tiffFileName=sys.argv[1]
#jsonFileName='viirsDistrictLevel/Census_2011/2011_Dist.geojson'
jsonFileName=sys.argv[2]

inputFolder=tiffFileName[:-4]
if not os.path.exists(inputFolder):
    os.makedirs(inputFolder)


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