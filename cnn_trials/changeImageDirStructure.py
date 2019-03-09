import math
import json
import sys
import pandas as pd
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
import shutil
import random
cutFilesFolder = sys.argv[1]
labelsFolder ='villageLabels'
villageLabelsFilesList = [f for f in listdir(labelsFolder) if (isfile(join(labelsFolder, f)) and f.endswith('.csv')) ]
print(villageLabelsFilesList)
predictionColumnDict={
    'VillageLabels_FC.csv':'Village_HHD_Cluster_FC',
    'VillageLabels_EMP.csv':'Village_HHD_Cluster_EMP',
    'VillageLabels_BF.csv':'Village_HHD_Cluster_BF',
    'VillageLabels_CHH.csv':'Village_HHD_Cluster_CHH',
    'VillageLabels_MSW.csv':'Village_HHD_Cluster_MSW',
    'VillageLabels_MSL.csv':'Village_HHD_Cluster_MSL',
}

villageCutFilesList = [f for f in listdir(cutFilesFolder) if isfile(join(cutFilesFolder, f))]

def getVcode(myStr):
    indexOfStart=myStr.rfind('@')+1
    return int(myStr[indexOfStart:])
    
listVcodeInImages=[]
village_File_Dictionary={}
vid_filename_dictionary={}
for tempStr in villageCutFilesList:
    splitUpList=tempStr[:-4].split('@')
    splitUpList=splitUpList[1:]
    keyVillage = '@'.join(splitUpList)
    cVcode=getVcode(keyVillage)
    listVcodeInImages.append(cVcode)
    vid_filename_dictionary[cVcode]=tempStr
    if (not (keyVillage in village_File_Dictionary)):
        village_File_Dictionary[keyVillage]=[tempStr]
    else:
        village_File_Dictionary[keyVillage].append(tempStr)
        
        
print(len(villageCutFilesList))
print(len(village_File_Dictionary))

for z in villageLabelsFilesList:
    print(z)
    indexOfStart_z=z.rfind('_')+1
    indicatorString=z[indexOfStart_z:][:-4]
    print('indicatorString',indicatorString)
    columnToPredict=predictionColumnDict[z]
    currLabelCsv = labelsFolder +'/'+z
    print(currLabelCsv)
    currLabelCsvContents = pd.read_csv(currLabelCsv)
    #print(currLabelCsvContents.head())
    tempVid=currLabelCsvContents['Town/Village'].values
    currLabelCsvContents=currLabelCsvContents[currLabelCsvContents['Town/Village'].isin(listVcodeInImages)]
    print(currLabelCsvContents.shape)
    folderTrain=cutFilesFolder+'_'+indicatorString+'_train'
    folderTest=cutFilesFolder+'_'+indicatorString+'_test'
    os.makedirs(folderTrain, exist_ok=True)
    os.makedirs(folderTest, exist_ok=True)
    #makeTheseFolders
    folderTrain_classFolder=[folderTrain+"/"+"level_"+str(i) for i in range(3)]
    folderTest_classFolder=[folderTest+"/"+"level_"+str(i) for i in range(3)]
    for zx in folderTest_classFolder:
        os.makedirs(zx, exist_ok=True)
    for zx in folderTrain_classFolder:
        os.makedirs(zx, exist_ok=True)
        
    for pqr in range(1,4):
        currLabelCsvContents_copy=currLabelCsvContents.copy()
        currLabelCsvContents_copy=currLabelCsvContents_copy[currLabelCsvContents_copy[columnToPredict]==pqr]
        percentTest = 0.2
        vidValuesCurrent=currLabelCsvContents_copy['Town/Village'].values
        random.shuffle(vidValuesCurrent)
        numPointsInTest = int(percentTest*len(vidValuesCurrent))
        testVids=(vidValuesCurrent[0:numPointsInTest])
        trainVids = (vidValuesCurrent[numPointsInTest:])
        currentPasteFolderTrain=folderTrain_classFolder[pqr-1]
        currentPasteFolderTest=folderTest_classFolder[pqr-1]

        for villageIdz in testVids:
            cvxFile=vid_filename_dictionary[villageIdz]
            shutil.copy(cutFilesFolder+'/'+cvxFile,currentPasteFolderTest+'/'+cvxFile)

        for villageIdz in trainVids:
            cvxFile=vid_filename_dictionary[villageIdz]
            shutil.copy(cutFilesFolder+'/'+cvxFile,currentPasteFolderTrain+'/'+cvxFile)
    
    print("="*30)