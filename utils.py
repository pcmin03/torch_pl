# load sample data
import json , re,random
import torch 
import numpy as np 
import cv2
import seaborn as sns 
import matplotlib 

from sklearn.metrics import confusion_matrix
from natsort import natsorted
from tqdm import tqdm 
from glob import glob 
import os 
import xml.etree.ElementTree as ET

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

    categories = ['Nontumor','score0','score1+','score2+','score3+']
    numlabel = np.zeros((5,1))
    clean_matric = np.zeros((len(categories),len(categories)))

    locat,countlab = np.unique(label,return_counts=True)    

    for i,j in zip(locat,countlab):
        numlabel[i] = j
    
    clean_matric += confusion_matrix(label,pred)
    matric = (clean_matric / numlabel) * 100 # change 100 precentage
    
    
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
    # oversampling by number of class
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

def get_coordination(jsonlabels): 
    # it work only work qupath label !!!!
    # rather will add phlips sdk 
    total_coordi = []
    for label in jsonlabels: 
        json_data=json.load(open(label))
        coordi = []
        for rois in json_data: 
            coordi.append(rois['geometry']['coordinates'])
        total_coordi.append(coordi)
    total_coordi = np.array(total_coordi)
    return total_coordi 

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

def makecolorspace():
    colorscore = [0,1,2,3,4]
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

def make_wsi(patchlist,roipath,wsin,roin,score,path):
        
    # load roi location 
    sample_label =  natsorted(glob(f'{roipath}/*.json'))
    mainpath = '/nfs3/cmpark/amc_erprher2'
    filepath=glob(f'{mainpath}/tiff/*3/')
    #load tiff images    
    sample_label = natsorted(glob(f'{roipath}*/*.xml'))
    print(sample_label)
    
    total_corrdi,_,_,_ = RegionPoint(sample_label[wsin])
    # total_corrdi = get_coordination(sample_label)[wsin][roin]
    bbox = makeBbox(np.array(total_corrdi[roin]).astype(int).squeeze())
    colordic = makecolorspace()
    
    # recovery patch image 
    # h,w = bbox[2:] - bbox[:2]
    # zerobox = np.empty([w,h,3]).astype(np.int32)
    # copyzerobox = zerobox.copy()
    # heatmapzerobox = zerobox.copy()
    
    alpha = 0.8
    # put original image 
    maxy, maxx = 0,0
    for patch in patchlist:
        locy,_,locx = np.array(re.split('[_XY.]',str(patch))[-4:-1])
        
        if maxy < int(locy): 
            maxy = int(locy)
        if maxx < int(locx): 
            maxx = int(locx)
    x,_,_ = cv2.imread(str(patch)).shape
    
    zerobox = np.empty([(maxx+1)*x,(maxy+1)*x,3]).astype(int)
    copyzerobox = zerobox.copy()
    heatmapzerobox = zerobox.copy()
    print(zerobox.shape,'asdasdaqsdasdasda')
    
    for patch,c in tqdm(zip(patchlist,score)) : 
        img = cv2.imread(str(patch))
        x,_,_ = img.shape
        locy,_,locx = np.array(re.split('[_XY.]',str(patch))[-4:-1])
        zerobox[x*int(locx):x*int(locx)+x,x*int(locy):x*int(locy)+x] = img
        copyzerobox[x*int(locx):x*int(locx)+x,x*int(locy):x*int(locy)+x] = colordic[c]
        colorimg = copyzerobox[x*int(locx):x*int(locx)+x,x*int(locy):x*int(locy)+x].copy()
        balndimg = (img * alpha) + (colorimg * (1-alpha)) 
        balndimg = balndimg.astype(np.uint8)
        heatmapzerobox[x*int(locx):x*int(locx)+x,x*int(locy):x*int(locy)+x] =  balndimg
    # make result dir 
    save_path = f'{path}/result'
    if not os.path.exists(save_path): 
        os.mkdir(save_path)
    print(save_path,'asdfasdfasdfasd')
    print(cv2.imwrite(f'{save_path}/sample.png',zerobox))
    cv2.imwrite(f'{save_path}/sample.png',zerobox)
    cv2.imwrite(f'{save_path}/colormap2.png',copyzerobox)
    cv2.imwrite(f'{save_path}/heatmap.png',heatmapzerobox)
    print('save!!!!')

import pandas as pd
def savecsv(sample:list,path): 
    if not os.path.exists(path): 
        os.mkdir(path)
    print(pd.DataFrame(sample))
    pd.DataFrame(sample).to_csv('result.csv')

def RegionPoint(roidir,returnnum=True): 
    meta_anno,allstye,numroi = [],[],0
    total_uid = []
    #load sample roidir
    # it return list 
    tree = ET.parse(roidir)
    root = tree.getroot() 
    Resion=root.findall('Regions/Region')
    
    for resion in Resion: 
        if type(resion.find('Uid').text) == str:
            uid = resion.find('Uid').text
            stype =resion.find('Type').text

            if stype == 'ClosedFreeform':
                points = [list(map(float,i.text.split(','))) for i in resion.findall('Points/Point')]
            elif stype == 'Rectangle':
                x1,y1 = list(map(float,resion.find('TopLeft').text.split(',')))
                x2,y2 = list(map(float,resion.find('BottomRight').text.split(',')))
                # make bbox
                points = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                
            elif stype == 'Ellipse':   
                axis11 = list(map(float,resion.find('Axis1/Point1').text.split(',')))
                axis12 = list(map(float,resion.find('Axis1/Point2').text.split(',')))
                axis21 = list(map(float,resion.find('Axis2/Point1').text.split(',')))
                axis22 = list(map(float,resion.find('Axis2/Point2').text.split(',')))
                
                points = [axis11,axis21,axis12,axis22]

            elif stype == 'Textbox': 
            
                continue
            
            meta_anno.append(points)
            allstye.append(stype)
            total_uid.append(uid)
            numroi += 1

    if returnnum == True: 
        return np.array(meta_anno),allstye , numroi,total_uid
    else:
        return np.array(meta_anno)

#slice patch size image
def makeBbox(point):
    x,y=point[:,0],point[:,1]
    x1,y1,x2,y2 = min(x),min(y),max(x),max(y)
    return np.array([x1,y1,x2,y2])

def recover_wsi(patchlist,roipath,location,score,path):

    def make_blandimg(images,scores): 
        colordic = makecolorspace()
        colorimg = np.zeros_like(images).astype(np.uint8)
        blendimg = colorimg.copy()
        alpha = 0.8
        
        for n in range(len(images)): 
            
            colorimg[n] = colordic[scores[n]]
            blendimg[n] = (images[n] * alpha) + (colorimg[n] * (1-alpha)) 

        return blendimg,colorimg
        
    xl = location[:,0].max()+1
    yl = location[:,1].max()+1

    
    print(patchlist.shape,'232323')
    # patchlist = np.array(list(map(lambda x : x[:,:,::-1],patchlist)))
    patchlist = np.array(list(map(lambda x : x[64:192,64:192][:,:,::-1],patchlist))) # crop center point 128 x 128 
    print(patchlist.shape)

    hstackimg,hstackcolor,hstackbland  = [],[],[]
    for i in range(yl): 
        patchimages = patchlist[(i*xl):(i+1)*xl]
        blendimg, colorimg = make_blandimg(patchimages,score[(i*xl):(i+1)*xl])

        hstackimg.append(np.hstack(patchimages))
        hstackcolor.append(np.hstack(colorimg))
        hstackbland.append(np.hstack(blendimg))
    
    hstackimg = np.vstack(np.array(hstackimg))
    hstackcolor = np.vstack(np.array(hstackcolor))
    hstackbland = np.vstack(np.array(hstackbland))

    save_path = f'{path}/result'
    if os.path.exists(save_path): 
        os.makedirs(save_path,exist_ok=True)

    cv2.imwrite(f'{save_path}/stackimg.png',hstackimg)
    cv2.imwrite(f'{save_path}/stackcolor.png',hstackcolor)
    cv2.imwrite(f'{save_path}/stackbland.png',hstackbland)
    print('save!!!!')

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    binary_mask = np.where(mask > 0.5, mask,np.zeros_like(mask))
    satck_mask = np.stack((binary_mask,binary_mask,binary_mask),axis=-1)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    
    cv2.imwrite('sampelasdas.png',np.uint8(heatmap*satck_mask*255))

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")
    cv2.imwrite('img.png',np.uint8(img*255))
    cam = cv2.add(heatmap*satck_mask*255,img*255)
    # cam = cam / np.max(cam)
    return np.uint8(cam)


if __name__ == "__main__":
    sample = './sample.json'
    json2label(sample)