import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST,CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import albumentations as A
import torchvision.models as models
import numpy as np
import random 
import torchvision

from pytorch_lightning import loggers as pl_loggers

class HER2classify(pl.LightningModule): 
    def __init__(self,model): 
        super().__init__()
        self.encoder = model
    
#     def forward(self,x): 
        
        
    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
    
    def training_step(self,batch, batch_idx): 
        x,y  = batch 
        pred = self.encoder(x)
        loss = F.cross_entropy(pred,y)
        self.log('train_loss',loss)
        return loss 
    
    def validation_step(self,batch, batch_idx): 
        x,y  = batch 
        pred = self.encoder(x)
        loss = F.cross_entropy(pred,y)
        self.log('val_loss',loss)
        return loss 

    def test_step(self,batch, batch_idx): 
        x,y  = batch 
        pred = self.encoder(x)
        loss = F.cross_entropy(pred,y)
        self.log('test_loss',loss)
        return loss 
    
        
# random seed
def resetseed(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)

if __name__ == "__main__":
    
    #define model 
    resent18 = models.resnet18(True)
    
    ## sample cifar 10 
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 500

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # start train 

    resetseed(2021)

    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    HER2net = HER2classify(resent18)
    trainer = pl.Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=20,logger=tb_logger)
    trainer.fit(HER2net,trainloader,testloader)
