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
trainImgFolder=sys.argv[1]
checkPtFolder=trainImgFolder[:-1]+'_checkpoints'
os.makedirs(checkPtFolder, exist_ok=True)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 1000
num_classes = 2
batch_size = 100
learning_rate = 0.001

myTransform = transforms.Compose(
                   [transforms.Resize((64,64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.65, 0.65, 0.65), (0.15, 0.15, 0.15))])


train_dataset = torchvision.datasets.ImageFolder(root=trainImgFolder,
                                                transform=myTransform)


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           sampler=ImbalancedDatasetSampler(train_dataset))


model = models.resnet18(num_classes=num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    file_name=checkPtFolder+'/'+'epoch_'+str(epoch)+'.ckpt'
    torch.save(model.state_dict(), file_name)

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)