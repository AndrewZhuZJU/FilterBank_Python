# -*- coding: utf-8 -*-

###########################################
# Created by andrew
###########################################

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#data_dir = 'Assignment1_data/data/'
data_dir = 'Assignment1_data/query/'
image_dataset = datasets.ImageFolder(os.path.join(data_dir),data_transforms)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=32,
                                             shuffle=False, num_workers=4)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
model= models.resnet101(pretrained=True)
#model= models.resnet50(pretrained=True)

model = nn.DataParallel(model)
model = model.to(device)
modules = list(model.children())[0]
modules = list(modules.children())[:-1]
FeatureExtractor = nn.Sequential(*modules)
FeatureExtractor.eval()
resFea = []
for i, imgs in enumerate(dataloader):
    print(i)
    img = imgs[0].to(device)
    outputs = FeatureExtractor(img).view(-1,2048)
    print(outputs.shape)
    if(not i):
        resFea = outputs.detach().cpu().numpy()
    else:
        resFea = np.concatenate((resFea,outputs.detach().cpu().numpy()))

#np.save('Res50Features.npy',resFea)
np.save('QueryRes101Features.npy',resFea)