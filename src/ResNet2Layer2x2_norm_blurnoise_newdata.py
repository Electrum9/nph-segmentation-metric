#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from PIL import Image
import nibabel as nib
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[3]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# In[4]:


def read_txt(txt_path):
    
    with open(txt_path) as f:
        lines = f.readlines()
    
    return lines
def read_patch(txt_path):
    txt_data = []
    lines=read_txt(txt_path)
    for line in lines:
        lineData=line.strip().split(',')
        txt_data.append([lineData[0],lineData[1],lineData[2],int(lineData[3]),int(lineData[4]),int(lineData[5])])
            
    return txt_data

def read_pos(txt_path):
    txt_data = {}
    lines=read_txt(txt_path)
    for line in lines:
        lineData=line.strip().split(',')
        txt_data[lineData[0]]=[int(lineData[1]),int(lineData[2]),int(lineData[3])]
            
    return txt_data
    

def getPatch(imPath, segPath, i, j, k):
    
    with open(imPath, 'rb') as f:

        image = np.load(f)

    with open(segPath, 'rb') as f:

        segmentation = np.load(f)[1:,1:]
        
        
        
    for x in range(segmentation.shape[0]):
        for y in range(segmentation.shape[1]):
            if segmentation[x,y]==6 or segmentation[x,y]==3:
                segmentation[x,y]=1
            
            if segmentation[x,y]==5:
                segmentation[x,y]=2

            if segmentation[x,y]==4:
                segmentation[x,y]=3
    
    
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
                
                if image[x,y,z]>200: image[x,y,z]=200
                if image[x,y,z]<-100: image[x,y,z]=-100
    
    image+=100
    image=image/300
    
    
    return image, segmentation, torch.tensor([i,j,k])    


# In[5]:

    
    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)    

    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        
        inputsNew = inputs.view(-1)
#         print(inputsNew)
        targetsNew=F.one_hot(targets.to(torch.int64), 4).float()
#         print(targetsNew)
        targetsNew = targetsNew.view(-1)
#         print(targetsNew)
        
        intersection = (inputsNew * targetsNew).sum()                            
        dice = (2.*intersection + smooth)/(inputsNew.sum() + targetsNew.sum() + smooth)  
        
        return 1 - dice


class NPHDataset(Dataset):
    def __init__(self, dataPath, segPath, txtPosition, Train=False):
      
        self.dataPath=dataPath
        self.segPath=segPath
        self.txtPosition = txtPosition
        self.train=Train
        self.imgList=read_patch(txtPosition)
        
        self.shapeList=read_pos('data-split/image_shape.txt')
        
        if Train:
            self.transform=transforms.Compose([
                    transforms.ToTensor(),

                    transforms.GaussianBlur(3, sigma=(36/300*0.05)),
                    AddGaussianNoise(0., 36/300*0.05)

            ])
        
        else:
            self.transform=transforms.ToTensor()
        
        
        
    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        volName=self.imgList[idx][0]
        imageName=self.imgList[idx][1]
        segName=self.imgList[idx][2]
        

        imgPath = os.path.join(self.dataPath, imageName)
        segPath = os.path.join(self.segPath, segName)
        data, annotation, pos=getPatch(imgPath, segPath, 
                                    self.imgList[idx][3],self.imgList[idx][4],self.imgList[idx][5])

        shape=self.shapeList[volName]
   
        image = self.transform(data.copy())
        sample = {'img': image,
                  'label': annotation.copy(),
                  'pos': pos,
                  'name': volName,
                  'shape':np.array(shape)
                 }
        return sample




BS=200 
# BASE='/home/fei/documents/GitHub/NPH_Segmentation/data-split'

BASE='data-split'
dataPath=os.path.join(BASE,'Scans_patch_mixed_6')
segPath=os.path.join(BASE,'Segmentation_patch_mixed_6')


trainDataset=NPHDataset(dataPath,segPath,os.path.join(BASE,'train_positions_mixed_6.txt'),Train=True)
train_loader = DataLoader(trainDataset, batch_size=BS, num_workers=8, prefetch_factor=10000, pin_memory=True, drop_last=False, shuffle=True, persistent_workers=True)

valDataset=NPHDataset(dataPath,segPath,os.path.join(BASE,'val_positions_mixed_6.txt'),Train=False)
val_loader = DataLoader(valDataset, batch_size=BS, num_workers=8, prefetch_factor=5000, pin_memory=True, drop_last=False, shuffle=True, persistent_workers=True)

testDataPath=os.path.join(BASE, 'Scan_patch_test')
testSegPath=os.path.join(BASE, 'Segmentation_patch_test')

testDataset=NPHDataset(testDataPath,testSegPath,os.path.join(BASE,'test_positions_rand.txt'),Train=False)
test_loader = DataLoader(testDataset, batch_size=BS, num_workers=8, prefetch_factor=5000, pin_memory=True, drop_last=False, shuffle=True, persistent_workers=True)


# 
ResNet=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
class MyModel(nn.Module):
    def __init__(self,ResNet, num_classes=4, num_outputs=9):
        super(MyModel, self).__init__()
 
        self.layer0=nn.Sequential(
            nn.Conv2d(3,64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False),
            
        )
        
        self.layer1=ResNet.layer1
        self.layer2=ResNet.layer2

        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.fc=nn.Linear(in_features=128, out_features=num_classes*num_outputs, bias=True)
        
    def forward(self, x):

        x=self.layer0(x)
        x=self.layer1(x)        
        x=self.layer2(x) 

        x=self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)              

        return x
model = MyModel(ResNet,num_classes=4, num_outputs=4).to(device)
model.load_state_dict(torch.load('model_backup/epoch49_ResNet2D3Class_wd6_2Layer2x2_300.pt',map_location=device))

def dice(TP, FN, FP, pred, target):
    correct=0
    for i in range(pred.shape[1]):
        for j in range(pred.shape[2]):
            if pred[0, i,j].view_as(target[i,j])==target[i,j]:
                TP[int(target[i,j])]+=1

                correct+=1

            else:
                FN[int(target[i,j])]+=1
                FP[int(pred[0, i,j])]+=1
    return correct



def evaluation(output, target, TP, FP, FN):
    total=0
    correct=0
    
    criteria = nn.CrossEntropyLoss()
#     criteria=DiceLoss()
    loss=criteria(output, target.long())        
    pred=output.argmax(dim=1, keepdim=True)    
    N=output.shape[0]
    for k in range(N):

        correct+=dice(TP, FN, FP, pred[k,:,:,:], target[k,:,:])
    
        total+=4

    return loss, correct, total


# In[9]:


def train(optimizer, epoch):
    
    bs=BS
    model.train()

    trainCorrect=0
    trainTotal=0
    trainLoss=0
    TP=[0]*7
    FP=[0]*7
    FN=[0]*7

    
    for batch_index, batch_samples in enumerate(train_loader):

        # move data to device
        data, target = batch_samples['img'].to(device, dtype=torch.float), batch_samples['label'].to(device)
        
        pos, shape=batch_samples['pos'].to(device, dtype=torch.float), batch_samples['shape'].to(device)
        optimizer.zero_grad()

        output = model(data)
        softmax=nn.Softmax(dim=1)
        output=torch.reshape(output,(output.shape[0], 4, 2,2))
        output=softmax(output)

        loss, correct, total= evaluation(output, target, TP, FP, FN)
        
        loss.backward()
        optimizer.step()
        
        trainCorrect+=correct
        trainLoss+=loss
        trainTotal+=total


        if (batch_index+1) % (200) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f} Current accuracy: {:.3f}% '.format(
                epoch, batch_index+1, len(train_loader),
                100.0 * batch_index / len(train_loader), trainLoss.item()/(batch_index+1), trainCorrect/trainTotal*100))
            

    print('Train Epoch: {}, Correct point: {}/{}'.format(epoch, trainCorrect, trainTotal))   
    for i in range(1,4):
        print('    Dice score for class{}: {}'.format(i, 2*TP[i]/(2*TP[i]+FP[i]+FN[i])))

    
    return trainLoss, trainCorrect, trainTotal, TP, FN, FP


# In[10]:


def test(epoch, test_loader, status):
    
    model.eval()
    testLoss = 0
    testCorrect = 0
    testTotal=0
    bs=BS
    TP=[0]*7
    FP=[0]*7
    FN=[0]*7

    result=[]
    # Don't update model
    with torch.no_grad():
        predList=[]
        targetList=[]
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to(device, dtype=torch.float), batch_samples['label'].to(device)
            pos, shape=batch_samples['pos'].to(device, dtype=torch.float), batch_samples['shape'].to(device)

            output = model(data)
            softmax=nn.Softmax(dim=1)
            output=torch.reshape(output,(output.shape[0],4, 2,2))
            output=softmax(output)

            loss, correct, total = evaluation(output, target, TP, FP, FN)
            testCorrect+=correct
            testLoss+=loss
            testTotal+=total

            if (batch_index+1) % (100) == 0:
                print('{} Epoch: {} [{}/{} ({:.0f}%)]\tTest Loss: {:.6f} Current accuracy: {:.3f}%'.format(status,
                    epoch, batch_index+1, len(test_loader),
                    100.0 * batch_index / len(test_loader), testLoss.item()/(batch_index+1), testCorrect/testTotal*100))

                
    print('{} Epoch {}: Correct point: {}/{}, {}'.format(status, epoch, testCorrect, testTotal, testCorrect/testTotal*100))   
    for i in range(1,4):
        print('    Dice score for class{}: {}'.format(i, 2*TP[i]/(2*TP[i]+FP[i]+FN[i])))

    
    return testLoss, testCorrect, testTotal, TP, FN, FP


# In[11]:


optimizer =optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)
modelname='3Class_wd6_2Layer2x2_300'
numEpoch=100

for i in range(50,numEpoch):

    trainLoss, trainCorrect, trainTotal, TP, FN, FP=train(optimizer, i)
    with open('results/result_data_{}.txt'.format(modelname),"a+") as file:
        file.write('Epoch {}, train loss: {}, total: {},correct: {}, accuracy: {:.6f}\n'.format(i, trainLoss, trainTotal, trainCorrect, trainCorrect/trainTotal))
    with open('results/result_classwise_{}.txt'.format(modelname),"a+") as file:
        for j in range(1,4):
            file.write('Epoch {}, train dice score for class{}: {}\n'.format(i, j, 2*TP[j]/(2*TP[j]+FP[j]+FN[j])))
    
    valLoss, valCorrect, valTotal, TP, FN, FP=test(i, val_loader, 'Val')
    with open('results/result_data_{}.txt'.format(modelname),"a+") as file:
        file.write('Epoch {}, val loss: {}, total: {},correct: {}, accuracy: {:.6f}\n'.format(i, valLoss, valTotal, valCorrect, valCorrect/valTotal))
    
    with open('results/result_classwise_{}.txt'.format(modelname),"a+") as file:
        for j in range(1,4):
            file.write('Epoch {}, val dice score for class{}: {}\n'.format(i, j, 2*TP[j]/(2*TP[j]+FP[j]+FN[j])))

    testLoss, testCorrect, testTotal, TP, FN, FP=test(i, test_loader, 'Test')
    with open('results/result_data_{}.txt'.format(modelname),"a+") as file:
        file.write('Epoch {}, test loss: {}, total: {},correct: {}, accuracy: {:.6f}\n'.format(i, testLoss, testTotal, testCorrect, testCorrect/testTotal))

    with open('results/result_classwise_{}.txt'.format(modelname),"a+") as file:
        for j in range(1,4):
            file.write('Epoch {}, test dice score for class{}: {}\n'.format(i, j, 2*TP[j]/(2*TP[j]+FP[j]+FN[j])))

            
            
        
    if (i)%5==0: 
        torch.save(model.state_dict(), "model_backup/epoch{}_2Dresnet{}.pt".format(i, modelname))  
        


torch.save(model.state_dict(), "model_backup/epoch{}_ResNet2D{}.pt".format(i, modelname)) 

