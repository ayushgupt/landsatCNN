import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
from os import listdir
from os.path import isfile, join
from sklearn.metrics import f1_score
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


mypath='normal_csvFiles/normal_csv/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# print(onlyfiles)
labels_2011=pd.read_csv('2011_label.csv')
array1=(labels_2011['EMP_2011'].values)
df_emp_to_number={'Unemployment':1,'Agricultural':2,'Non Agri':3}
array2=np.array([df_emp_to_number[t] for t in array1])
labels_2011['numEMP_2011']=array2
trainCols=['light_'+str(i) for i in range(0,101,1)]
predictCols=['MSW_2011','BF_2011','MSL_2011','FC_2011','CHH_2011','numEMP_2011']


acc_df = pd.DataFrame(columns=('Month','MSW_2011','BF_2011','MSL_2011','FC_2011','CHH_2011','numEMP_2011'))
loc_int=0
for csvFileName1 in onlyfiles:
    rowToAppend=[csvFileName1[:-4]]
    print('For Month: ', csvFileName1)
    #csvFileName='corrected_csvFiles/corrected_csv/corrected_viirs_India_2014-01-01_2014-01-31_500.csv'
    csvFileName=mypath+csvFileName1
    viirsBucketsData=pd.read_csv(csvFileName)
    viirsBucketsData.rename( columns={'Unnamed: 0':'dname'}, inplace=True)
    combinedDf = pd.merge(viirsBucketsData, labels_2011, left_on=['censuscode'], right_on = ['District'])
    for prediction_label in predictCols: 
        #print(prediction_label)
        
        list_acc=[]
        for i in range(5):
            train_features, test_features, train_labels, test_labels = train_test_split(combinedDf[trainCols], 
                                                                                        combinedDf[prediction_label],
                                                                                        test_size = 0.3,
                                                                                        random_state = int(random.random()))
            
            scaler = StandardScaler()
            scaler.fit(train_features)
            rf = RandomForestClassifier(n_estimators = 500, random_state = 64, class_weight='balanced')
            rf.fit(scaler.transform(train_features), train_labels)
            predictions = rf.predict(scaler.transform(test_features))
            score_curr=f1_score(test_labels, predictions, average='weighted')
            list_acc.append(score_curr)
            
        rowToAppend.append(mean(list_acc))    
        
    acc_df.loc[loc_int] = rowToAppend
    loc_int+=1



acc_df.to_csv('accuracy_normal_viirs_balanced.csv')