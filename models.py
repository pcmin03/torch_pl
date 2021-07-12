import torchvision.models as torch_models
from torch import nn 
from efficientnet_pytorch import EfficientNet
import timm 

def resnet(out_ch,pretrain,modelname:str): 
    if modelname =='resnet18': 
        models = torch_models.resnet18(pretrain)
    if modelname =='resnet34': 
        models = torch_models.resnet34(pretrain)
    if modelname =='resnet50': 
        models = torch_models.resnet50(pretrain)

    models.fc = nn.Linear(512,out_ch)
    return models

def resnetcam(out_ch,pretrain,modelname:str): 
    models = torch_models.resnet18(pretrain)
    # model.fc

def efficentnet(out_ch,pretrain,modelname:str): 
    # modelname efficientnet-b0 ~ b7
    print(modelname,'asdfasdfasdfasd')
    if pretrain == True: 
        model = EfficientNet.from_pretrained(modelname)
    else : 
        model = EfficientNet.from_name(modelname)
    model = nn.Sequential(
            model,
            nn.Linear(1000,out_ch)
            )
    return model 


def choosemodel(out_ch,pretrain:bool,modelname:str): 
    
    if modelname.find('resnet') != -1: 
        return resnet(out_ch,pretrain,modelname)
    elif modelname.find('efficient') != -1: 
        return efficentnet(out_ch,pretrain,modelname)
    # else: 
    #     return timm.create(model('modelname'))
