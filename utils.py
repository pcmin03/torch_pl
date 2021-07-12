# load sample data
import json 
import torch 
import numpy as np 
import random 
import cv2
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from natsort import natsorted
def json2label(filepath): 
    total_label = []
    meta = json.load(open(filepath))
    for n,data in enumerate(meta):
        try :
            label = data['completions'][0]['result'][0]['value']['choices'][0]
            labelname = data['data']['image'].split('/')[-1]
            labelname = labelname.split('-')[-1]
            total_label.append([label,labelname])
        except :
            print(f'Empty label:{n}')
    return total_label

def denormalize(images): 

    max_value = 255.
    
    mean = np.array([0.80107421, 0.80363828, 0.80619713]) * max_value
    std = np.array([0.0785173 , 0.08986283, 0.09588921]) * max_value
    
    if images.ndim == 4: 
        images = (images.transpose((0,2,3,1)) * std) + mean 
        images = images[:,:,:,::-1]
        
        return images.astype(np.int32)

    elif images.ndim == 3: 
        images = images.transpose((1,2,0)) * std + mean 
        images = images[:,:,::-1]
        
        return images.astype(np.int32)
def confusionmatric(label,pred): 
#     print(label,pred,'2323232323')
#     classn = np.unique(label)
    categories = ['Nontumor','score0','score1+','score2+','score3+']
    _,numlabel = np.unique(label,return_counts=True)
    clean_matric = confusion_matrix(label,pred)
    matric = clean_matric / numlabel[:,None] * 100 # change 100 precentage
    ax = sns.heatmap(matric,cmap = 'YlGnBu',vmin=0, vmax=100,\
        xticklabels=categories,yticklabels=categories).get_figure()
    
    return ax,clean_matric

def calcuate_metric(matric):
    class_accurate = np.diag(matric)/ matric.sum(axis=1)   # accuacy
    class_specifi = np.diag(matric)/ np.sum(matric,axis=0) # precision
    class_sensiti = np.diag(matric)/ np.sum(matric,axis=1) # recall 

    # class_F1score = 2*((class_precision*class_recall)/(class_precision+class_recall))
    def make_dic(class_score): 
        matric_dic = {}
        for i in range(len(class_score)): 
            matric_dic[index2word(i)] = class_score[i] 
        return matric_dic

    total_score = {}
    score_name = ['acc','spec','sens']
    
    for n,i in enumerate([class_accurate,class_specifi,class_sensiti]):
        
        total_score[score_name[n]] = make_dic(i)

    return total_score

def index2word(x):
    indexdic = {0:'Nontumor',1:'score0',2:'score1+',3:'score2+',4:'score3+'}
    return indexdic[x]

def word2index(x): 
    wordic = {'Nontumor':0,'score0':1,'score1+':2,'score2+':3,'score3+':4}
    return wordic[x]

def insertext(images,testlist):
    colors = [(255,0,0),(0,0,255)]
    org = [(10,25),(10,55)]

    testlist = list(map(index2word,np.array(testlist)))
    
    copy_img = images.copy()
    for n,i in enumerate(testlist): 
        copy_img = cv2.putText(copy_img,i,org[n],cv2.FONT_HERSHEY_SIMPLEX,\
        0.5,colors[n],1)
#     cv2.imwrite('sample32323232.png',copy_img)
    
    return copy_img

def repeatarray(dataset): 
    
    names,labels = np.unique(dataset[:,1],return_inverse=True)
    addition_img = []
    for i in range(len(names)): 
        if i == 2 : 
            continue 
        else:
            quotient = len(dataset[labels==2])//len(dataset[labels==i])
        addition_img.append(np.repeat(dataset[labels==i],quotient,axis=0))
    sample = np.array(addition_img)
    addition_img=np.concatenate(addition_img)
    ov_dataset = np.concatenate((dataset,addition_img))
    return ov_dataset
    # random seed

def makeBbox(point:list):
    x,y=point[:,0],point[:,1]
    x1,y1,x2,y2 = min(x),min(y),max(x),max(y)
    return np.array([x1,y1,x2,y2])


def resetseed(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
#load recovery image patch size 
# it need the poloygon corrdination information 
# it must have image location like R_{num}_X{num}_Y{num}
from pathlib import Path
import json 
import re
from tqdm import tqdm 
from glob import glob 
import matplotlib 

def makecolorspace():
    colorscore = [0,1,2,3,4]
    # cmap = matplotlib.cm.get_cmap('seismic')
    # x = np.linspace(0.0,1.0,len(colorscore))
    cmap = matplotlib.cm.get_cmap('seismic')
    neg = np.linspace(0.1,0.0,2)
    pos = np.linspace(1.0,0.7,2)
    non = [0.5]

    x = 1 - np.concatenate((non,neg,pos))

    colordic = {}
    for n in range(len(colorscore)):
        randv = [int(j*255) for j in cmap(x[n])[:3]]
        colordic[n] = randv

    return colordic


if __name__ == "__main__":
    sample = './sample.json'
    json2label(sample)