from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np 

def transforms(): 
    
    mean = [0.80107421, 0.80363828, 0.80619713]
    std = [0.0785173 , 0.08986283, 0.09588921]
    trans = {'fit':A.Compose([
                        
                        A.OneOf([ # simple annotation 
                        A.HorizontalFlip(p=1),
                        A.RandomRotate90(p=1),
                        A.Rotate(limit=30,p=1)],p=1),
                        
                        # A.OneOf([ # instance annotation
                        # A.CLAHE(p=0.5)
                        # A.Cutout(20,20,p=0.5)
                        # A.HueSaturationValue()
                        # ],p=1),
                    # A.RandomSizedCrop((128,128),256,256,p=1),
                    A.Normalize(mean,std),
                    ToTensorV2()
                            ]),
             
                'test': A.Compose([
                        # A.RandomSizedCrop((128,128),256,256,p=1),
                        # A.OneOf([ # simple annotation w/ TTA
                        # A.HorizontalFlip(p=1),
                        # A.RandomRotate90(p=1),
                        # A.Rotate(limit=30,p=1)],p=1),
                        
                        A.Normalize(mean,std), 
                        ToTensorV2()
                        ])}

    return trans

class HER2Dataset(Dataset): 
    def __init__(self,imagedir,stage:str): 
        self.transform = transforms()[stage]
        self.imgdir = imagedir
        self.stage = stage

    def __len__(self): 
        return len(self.imgdir) 

    def __getitem__(self,idx): 
        
        imgpath, label = self.imgdir[idx]
        image = cv2.imread(imgpath)[:,:,::-1]
        sample = self.transform(image=image)
        return sample['image'],int(label)

class inferHER2Dataset(Dataset): 
    def __init__(self,images,stage:str,npimg:bool): 
        self.transform = transforms()[stage]
        self.images = images
        self.stage = stage
        self.isimg = npimg 

    def __len__(self): 
        return len(self.images) 

    def __getitem__(self,idx): 
        if self.isimg == True: 
            image, xylocat = self.images[idx]
            sample = self.transform(image=image)
            return sample['image'],xylocat

        elif self.isimg == False: 
            image = cv2.imread(self.images[idx])[:,:,::-1]
            sample = self.transform(image=image)
            imagepath = self.images[idx]
            
            return sample['image'],imagepath


        