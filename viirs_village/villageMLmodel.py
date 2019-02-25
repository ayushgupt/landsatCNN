import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import time
import math
import json
import sys
import rasterio
from rasterio.mask import mask
from PIL import Image
from os import listdir
from os.path import isfile, join
import pickle
import h5py
import scipy.misc
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import random
from statistics import mean 

# Confusion matrix code
from sklearn.metrics import confusion_matrix
from itertools import product
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print('')
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
villageFeaturesCsv = sys.argv[1] #'data/2014_dec_cutFiles_trial.csv'
labelsFolder ='villageLabels'

resultsDir = villageFeaturesCsv[:-4]+'_results'


if not os.path.exists(resultsDir):
    os.makedirs(resultsDir)

    
f = open(resultsDir+'/'+"accuracyValues.txt", "w")

trainCols=['light_'+str(i) for i in range(0,129,1)]
onlyfiles = [f for f in listdir(labelsFolder) if (isfile(join(labelsFolder, f)) and f.endswith('.csv')) ]
# print(onlyfiles)
predictionColumnDict={
    'VillageLabels_FC.csv':'Village_HHD_Cluster_FC',
    'VillageLabels_EMP.csv':'Village_HHD_Cluster_EMP',
    'VillageLabels_BF.csv':'Village_HHD_Cluster_BF',
    'VillageLabels_CHH.csv':'Village_HHD_Cluster_CHH',
    'VillageLabels_MSW.csv':'Village_HHD_Cluster_MSW',
    'VillageLabels_MSL.csv':'Village_HHD_Cluster_MSL',
}

df1 = pd.read_csv(villageFeaturesCsv)
for z in onlyfiles:
    f.write(z+'\n')
    columnToPredict=predictionColumnDict[z]
    currLabelCsv = labelsFolder +'/'+z
    df = pd.read_csv(currLabelCsv)
    dfMerged=df1.merge(df, left_on='vCode2011', right_on='Town/Village')
    train_features, test_features, train_labels, test_labels = train_test_split(dfMerged[trainCols], 
                                                                                        dfMerged[columnToPredict],
                                                                                        test_size = 0.3,
                                                                                        random_state = int(random.random()))
            
    rf = RandomForestClassifier(n_estimators = 500, random_state = 64, class_weight='balanced')
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    f.write('accuracy_score::'+str(accuracy_score(test_labels, predictions))+'\n')
    f.write('f1_score::'+str(f1_score(test_labels, predictions,average='weighted'))+'\n')
    cnf_matrix = confusion_matrix(test_labels, predictions)
    class_names=['1. Under-Developed','2. Moderately-Developed','3. Developed']
    if(columnToPredict=='Village_HHD_Cluster_EMP'):
        class_names=['1. Unemployment','2. Agricultural','3. Non Agricultural Employment']
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title=columnToPredict+'_Normalized')
    #plt.show()
    plt.savefig(resultsDir+'/'+columnToPredict+'_Normalized.jpg')
    plt.clf()
    time.sleep(2)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,title=columnToPredict+'_Absolute')
    plt.savefig(resultsDir+'/'+columnToPredict+'_Absolute.jpg')
    #plt.show()
    plt.clf()