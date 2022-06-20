import imp
import os 
import numpy as np
import torch
import glob
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.autograd import Variable
import torchvision
import pathlib
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=(-20,+20)),
    transforms.RandomInvert(p=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor=50, p=0.5),
    transforms.RandomAutocontrast(p=0.5),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

training_path ='C:\\Users\\Nicolae PC\\PyTorch\\TrafficSigns_Classification\\data\\train'
valid_path ='C:\\Users\\Nicolae PC\\PyTorch\TrafficSigns_Classification\\data\\val'


train_loader = DataLoader(torchvision.datasets.ImageFolder(training_path, transform=transformer), batch_size=4, shuffle = True)
valid_loader = DataLoader(torchvision.datasets.ImageFolder(valid_path, transform=transformer), batch_size=4, shuffle = True)

root = pathlib.Path(training_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

print(classes)

model = models.efficientnet_b0(pretrained=False,progress=True).to(device=device)
model.classifier._modules['1'] = nn.Linear(1280, 4).to(device=device)
print(model)
# print('model on cuda: ', next(model.parameters()).is_cuda)


optimizer=Adam(model.parameters(),lr=0.001)
loss_function=nn.CrossEntropyLoss()

num_epochs = 50

train_count = len(glob.glob(training_path+'/**/*.png'))
valid_count = len(glob.glob(valid_path+'/**/*.png'))

train_acc= []
valid_acc = []

print(train_count, valid_count)

best_accuracy=0.0

for epoch in range(num_epochs):
    
    #Evaluation and training on training dataset
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    
    for i, (images,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
            
        optimizer.zero_grad()
        
        outputs=model(images)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        
        train_loss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))
        
    train_accuracy=train_accuracy/train_count
    train_loss=train_loss/train_count
    
    
    # Evaluation on validing dataset
    model.eval()
    
    valid_accuracy=0.0
    for i, (images,labels) in enumerate(valid_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
            
        outputs=model(images)
        _,prediction=torch.max(outputs.data,1)
        valid_accuracy+=int(torch.sum(prediction==labels.data))
    
    valid_accuracy=valid_accuracy/valid_count

    valid_acc.append(valid_accuracy)
    train_acc.append(train_accuracy)

    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Valid Accuracy: '+str(valid_accuracy))
    
    #Save the best model
    if valid_accuracy>best_accuracy:
        torch.save(model.state_dict(),'best_checkpoint_GTSRB.pt')
        best_accuracy=valid_accuracy

print(f"valid acc = {valid_acc}\n")
print(f"train acc = {train_acc}\n")
plt.figure(figsize=(10,5))
plt.title("Training and Validation accuracy")
plt.plot(valid_acc,label="val")
plt.plot(train_acc,label="train")
plt.xlabel("iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.show()