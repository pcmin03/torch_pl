import torchvision.models as torch_models
from torch import nn 

def resnet(out_ch,pretrain,layername:str): 
    if layername =='resnet18': 
        models = torch_models.resnet18(pretrain)
    if layername =='resnet34': 
        models = torch_models.resnet34(pretrain)
    if layername =='resnet50': 
        models = torch_models.resnet50(pretrain)
    models.fc = nn.Linear(512,out_ch)
    return models

def resnetcam(out_ch,pretrain,layername:str): 
    models = torch_models.resnet18(pretrain)
    model.fc