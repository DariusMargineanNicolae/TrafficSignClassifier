from tabnanny import check
from turtle import forward
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
import os
from PIL import Image
import pathlib
import glob
import cv2

training_path ='C:\\Users\\Nicolae PC\\PyTorch\\TrafficSigns_Classification\\data\\train'
testing_path ='C:\\Users\\Nicolae PC\\PyTorch\TrafficSigns_Classification\\data\\val'

root=pathlib.Path(training_path)
# classes=sorted([j.name.split('n')[-1] for j in root.iterdir()])
# print(classes)
classes=['give_way', 'no_entry', 'priority_road', 'stop']

model = models.efficientnet_b0(pretrained=False)
model.classifier._modules['1'] = nn.Linear(1280, 4)
checkpoint = torch.load('best_checkpoint_GTSRB.pt')
model.load_state_dict(checkpoint)
model.eval()

transformer=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

test_path = 'C:\\Users\\Nicolae PC\\PyTorch\\TrafficSigns_Classification\\data\\test'

# test_loader = DataLoader(torchvision.datasets.ImageFolder(test_path, transform=transformer), batch_size=4, shuffle = True)
# def test:
#     model.eval()
#     test_count = len(glob.glob(test_path+'/**/*.png'))
#     test_accuracy=0.0
#     for i, (images,labels) in enumerate(test_loader):
#         if torch.cuda.is_available():
#             images=Variable(images.cuda())
#             labels=Variable(labels.cuda())

#         outputs=model(images)
#         _,prediction=torch.max(outputs.data,1)
#         test_accuracy+=int(torch.sum(prediction==labels.data))

#         test_accuracy=test_accuracy/test_count
#     print('Test Accuracy:' +str(test_accuracy))

def prediction(img_path,transformer):
    
    image=Image.open(img_path).convert('RGB')
    
    image_tensor=transformer(image).float()
    
    
    image_tensor=image_tensor.unsqueeze_(0)
    
    if torch.cuda.is_available():
        image_tensor.cuda()
        
    input=Variable(image_tensor)
    
    
    output=model(input)
    
    index=output.data.numpy().argmax()
    
    pred=classes[index]
    return pred


# pred_path='C:\\Users\\Nicolae PC\\PyTorch\\TrafficSigns_Classification\\imgs'
# images_path=glob.glob(pred_path+'/*.png')

# pred_dict={}

# for i in images_path:
#     pred_dict[i] = prediction(i,transformer)

def return_value_of_a_dict(dict, key):
    value = dict.get(key)
    return value
# for x in pred_dict.keys():
#     print(return_value_of_a_dict(pred_dict, x))
# print(pred_dict)


