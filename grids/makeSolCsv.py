from libtiff import TIFF
import sys
import numpy as np
import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import pickle
import h5py
import scipy.misc
import pandas as pd

# inputFolder='apr_19_cutFiles'
inputFolder = sys.argv[1]
onlyfiles = [f for f in listdir(inputFolder) if isfile(join(inputFolder, f))]

nightSum={}
for currImageName in onlyfiles:
    destImageName=currImageName
    fullPath=inputFolder+'/'+currImageName
    tif = TIFF.open(fullPath, mode='r')
    image = tif.read_image()
    data = np.array(image)
    ndata = data[:,:,3]
    nightSum[currImageName] = [ndata.sum()/(64*64)]
    

sol_Dataframe=pd.DataFrame.from_dict(nightSum, orient='index',columns=['SOL'])
sol_Dataframe.to_csv(inputFolder+'_sol.csv')