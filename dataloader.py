import numpy as np 
import re 
from pathlib import Path 
from sklearn.model_selection import StratifiedKFold
from dataset import Dataset,inferDataset
from utils import json2label,word2index,repeatarray,get_coordination,makeBbox
from natsort import natsorted,index_natsorted
import albumentations as A

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from glob import glob
from tqdm import tqdm 

def stratifiedkfold(data,label,kfold,kfoldth): 
    images = np.array(data)
    skf = StratifiedKFold(n_splits=kfold,random_state=None,shuffle=False)
    for n,(train_index, valid_index) in enumerate(skf.split(images,label)):
        if n == kfoldth:
            trainset, validset= images[train_index], images[valid_index]
    return trainset, validset

def uniquename(images_path):
    wsith = list(map(lambda x : re.split('[/_]',x)[-4],images_path))
    _,wsith = np.unique(wsith,return_inverse=True)
    return wsith 

#--------------------lp data module version -----------------#
# method only required init, training_step, configure_optimizers
# however, it can use validation, test, model, inferece hook .. etc hook 


class dataloderpl(pl.LightningDataModule): 
    def __init__(self,data_dir:str,lab_path:str,classname:tuple,batchsize:int): 
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batchsize
        self.lab_path = lab_path
        self.classname = classname
        
    def setup(self,stage:str =None): 
        
        ## load image & label
        image_path = Path(self.data_dir)
        labels = np.array(json2label(self.lab_path))

        ## data prerpocessing remove the trash label
        labels = labels[index_natsorted(labels[:,1])]
        _,idx = np.unique(labels[:,0],return_inverse=True)

        ## filtering trash label 
        print(f'filtering label number : {len(labels[idx==1])}')
        labels = labels[idx!=1]
        ## convert one hot encoding
        labels[:,0] = list(map(word2index,labels[:,0]))

        ## combine path with label 
        images = np.array([[str(image_path/i[1]),i[0]] for i in labels]) #[path,label]
        ## Nested corss validation (stratfiedkfold data)
        wsith = uniquename(images[:,0]) #input filename
        self.trainset,self.validset = stratifiedkfold(images,wsith,3,0)
        
        wsith = uniquename(self.validset[:,0]) #input filename
        self.validset,self.testset = stratifiedkfold(self.validset,wsith,4,0)

        self.trainset= repeatarray(self.trainset)
        # self.validset= repeatarray(self.validset)
        # concnatnate data & label 
        if stage =='fit' or stage is None: 
            self.her2train = Dataset(self.trainset,'fit') 
            self.her2valid = Dataset(self.validset,'test') 
        if stage =='test'or stage is None:

            
            self.her2test = Dataset(self.testset,'test') 
            
    def train_dataloader(self):
        return DataLoader(self.her2train,shuffle=True,num_workers=8,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.her2valid,shuffle=False,num_workers=8,
                          batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.her2test,shuffle=False,num_workers=8,
                          batch_size=self.batch_size)
        
class inferdataloderpl(pl.LightningDataModule): 
    def __init__(self,data_dir:str,corr_path:str,wsin:int,roin:int,batchsize:int): 
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batchsize
        self.corr_path = corr_path
        self.wsin = wsin  # 0
        self.roin = roin  # 3
        # '/nfs3/cmpark/preprocessing_work/her2sample/20210607_256/'
        # '/nfs3/cmpark/her2sample/sampledata/BIGtifflabel'
    def setup(self,stage:str = None,useopenslide:bool=False): 

        patch_image_path = Path(self.data_dir)
        patch_image_list = natsorted(list(patch_image_path.glob('*')))
        patch_image_list = list(map(lambda x : list(x.glob('*')),patch_image_list))


        patchlist = patch_image_list[self.wsin]
        patchlist = np.array([ str(i) for i in patchlist if re.split('[/_.]',str(i))[-4] == f'R{self.roin}'])
        # print(patchlist)
        
        self.her2test = inferDataset(patchlist,'test',False) 
            
    def test_dataloader(self):
        return DataLoader(self.her2test,shuffle=False,num_workers=8,
                          batch_size=self.batch_size)
        

#--------------------torch moduel version -------------------#
if __name__=='__main__': 
    base_path = '/anypath/'
    image_path = Path(base_path)
    images = list(image_path.glob('*'))
    images = list(map(str,images))

    wsith = uniquename(images)
    trainset,validset = stratifiedkfold(images,uniquename(images),3,0)

    wsith = uniquename(validset)
    validset,testset = stratifiedkfold(validset,uniquename(validset),4,0)
    
    