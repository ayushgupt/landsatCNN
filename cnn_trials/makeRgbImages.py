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


inputFolder=sys.argv[1]
outputFolder=sys.argv[2]
os.makedirs(outputFolder, exist_ok=True)
onlyfiles = [f for f in listdir(inputFolder) if isfile(join(inputFolder, f))]


for currImageName in onlyfiles:
	# currImageName='kerela-0000000000-0000000000@10400.0@KL-113@627160.tif'
	destImageName=currImageName
	tif = TIFF.open(inputFolder+'/'+currImageName, mode='r')
	image = tif.read_image()
	data = np.array(image)
	scipy.misc.imsave(outputFolder+'/'+destImageName[:-4]+'.png', data)





