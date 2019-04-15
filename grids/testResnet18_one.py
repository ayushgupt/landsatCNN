from sampler import ImbalancedDatasetSampler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import torchvision.models as models
import sys
#trainImgFolder='data/bihar/bihar_2010_landsat7_cutFiles_rgb_BF_train/'
#testImgFolder='data/bihar/bihar_2010_landsat7_cutFiles_rgb_BF_test/'

#trainImgFolder = sys.argv[1]
testImgFolder = sys.argv[2]
modelFile = sys.argv[1]
#checkPtFolder=trainImgFolder[:-1]+'_checkpoints'
#os.makedirs(checkPtFolder, exist_ok=True)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 80
num_classes = 2
batch_size = 100
learning_rate = 0.001

myTransform = transforms.Compose(
                   [transforms.Resize((64,64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.65, 0.65, 0.65), (0.15, 0.15, 0.15))])


#train_dataset = torchvision.datasets.ImageFolder(root=trainImgFolder,
#                                                transform=myTransform)

test_dataset = torchvision.datasets.ImageFolder(root=testImgFolder,
                                               transform=myTransform)

# Data loader
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                           batch_size=batch_size,
#                                           sampler=ImbalancedDatasetSampler(train_dataset))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


model = models.resnet18(num_classes=num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



model = models.resnet18(num_classes=num_classes)
for tk in range(num_epochs):
    print('='*40)
    file_name=modelFile
    state_dict = torch.load(file_name)
    model.load_state_dict(state_dict)
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        trueLabels=[]
        predictedLabels=[]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            trueLabels.extend(labels.tolist())
            predictedLabels.extend(predicted.tolist())
        print(Counter(predictedLabels))
        print(Counter(trueLabels))
        predictedLabels = np.array(predictedLabels)
        trueLabels = np.array(trueLabels)
        for currentClass in range(3):
            print('='*20)
            maskClass=(predictedLabels==currentClass)
            print('class',currentClass,' accuracy_score: ',accuracy_score(predictedLabels[maskClass],trueLabels[maskClass]))
            print('='*20)
        print('f1_weighted',f1_score(trueLabels, predictedLabels, average='weighted'))
        print('f1_macro',f1_score(trueLabels, predictedLabels, average='macro'))
        print('f1_micro',f1_score(trueLabels, predictedLabels, average='micro'))
    break

