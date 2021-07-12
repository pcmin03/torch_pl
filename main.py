import json 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger


from utils import *
from HER2dataloader import dataloderpl , inferdataloderpl
from models import resnet
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import itertools
import torchvision
import cv2
import random 
from glob import glob

import os
class classify(pl.LightningModule): 
    def __init__(self,model,lr): 
        super().__init__()
        self.encoder = model
        self.lr = lr
        self.val_collector, self.train_collector = [] , []
        self.weight = torch.tensor([0.9, 0.76740331, 0.8       , 0.92320442, 0.93314917],dtype=torch.float32).to('cuda:0')
        # self.weight = torch.tensor([1,1,1,1,1],dtype=torch.float32).to('cuda:0')

        # checkpoint = torch.load()
    
    # inference check
    def forward(self,batch): 
        return self.encoder(batch[0])
    
    def evaluate(self,pred,y,phase):
        self.preds = torch.argmax(pred,dim=1)
        acc = accuracy(self.preds, y)
        self.log(f'{phase}_acc',acc)
        
    def plot_gridimg(self,images,target,pred,num,phase): 

        if num > 1 : 
            gridimg = []
            
            for i in range(num): 
                j = random.randint(0,len(images)-1)
                singleimg = denormalize(images[j])
                singleimg = insertext(singleimg,[target[j],pred[j]])
                gridimg.append(singleimg)

            singleimg =np.hstack(gridimg).transpose((2,0,1))[::-1,:,:] / 255.
            # singleimg = torchvision.utils.make_grid(singleimg)
        else: 
            singleimg = denormalize(images[num])
            singleimg = insertext(singleimg,[target[num],pred[num]]).transpose((2,0,1))[::-1,:,:] / 255.
        
        self.logger.experiment.add_image(f'{phase}_generated_images', singleimg, self.current_epoch)
        
    def plot_confusionmatric(self,collector,phase,allscore:bool): 

        ## plot confusion matric        
        collector = np.array(collector)
        y = list(itertools.chain(*itertools.chain(*collector[:,0])))
        preds = list(itertools.chain(*itertools.chain(*collector[:,1])))

        plt.figure(figsize = (10,7))
        conf_matric,np_matric = confusionmatric(y,preds)
        plt.close(conf_matric)
        self.logger.experiment.add_figure(f'{phase}_confusionmatric', conf_matric, self.current_epoch)
        self.collector = []

        if allscore == True: 
            sampel = calcuate_metric(np_matric)
            for key,value in sampel.items(): 
                # for name, val in value.items():
                # val =list(value.values())
                # for i in val : 
                for name,val in value.items():
                    self.logger.experiment.add_scalar(f'{phase}_{key}/{name}',val,self.current_epoch)
                # self.log(f'{phase}_{key}',value,self.current_epoch)

    def shared_step(self,batch,phase:str,evalu:bool): 
        x,y  = batch 
        pred = self.encoder(x)
        loss = F.cross_entropy(pred,y,weight=self.weight)
        
        if evalu == True: 
            self.evaluate(pred,y,phase)
        
        return loss
    
    
    def training_step(self,batch, batch_idx): 
        phase = 'train'
        loss =self.shared_step(batch,phase,True)
        self.log(f'{phase}_loss',loss, prog_bar=True, logger=True)
        
        al_batch = batch+[self.preds]
        _,y,preds = list(map(lambda x: x.detach().cpu().numpy(),al_batch))
        self.train_collector.append([[y],[preds]])
        
        return loss 

    def trainion_epoch_end(self,train_step_outputs): 
        # confusion matric
        self.plot_confusionmatric(self.train_collector,'train',True)
    
    def validation_step(self,batch, batch_idx): 
        phase='valid'

        loss = self.shared_step(batch,phase,True)   
        self.log(f'{phase}_loss',loss, prog_bar=True, logger=True)
        
        al_batch = batch+[self.preds]
        x,y,preds = list(map(lambda x: x.detach().cpu().numpy(),al_batch))
        
        self.plot_gridimg(x,y,preds,5,phase)
        self.val_collector.append([[y],[preds]])
        
        return loss 
    
    def validation_epoch_end(self,validation_step_outputs): 
        # confusion matric
        self.plot_confusionmatric(self.val_collector,'valid',True)
        
    def test_step(self,batch, batch_idx): 

        phase='test'
        
        preds = torch.argmax(self.forward(batch),dim=1).detach().cpu().numpy()
        # self.log(f'{phase}_loss',loss, prog_bar=True, logger=True)
        # x,y,preds = list(map(lambda x: x.detach().cpu().numpy(),al_batch))
        # self.plot_gridimg(x,y,preds,5,phase)
        
        self.val_collector.append([preds,batch[1].cpu().numpy()])
    
    def test_epoch_end(self,test_step_outputs): 
        # confusion matric
        # self.plot_confusionmatric(self.val_collector,'test',True)
        return self.val_collector

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        opt_sch = {
                'scheduler': ExponentialLR(optimizer, 0.99),
                'interval': 'step'  # called after each training step
                }    
        return {'optimizer':optimizer}
    
    # def configure_callbacks(self):
    #     early_stop = EarlyStopping(monitor='val_acc', mode="max")
    #     checkpoint = ModelCheckpoint(monitor='val_loss')
    #     return [early_stop, checkpoint]
    
if __name__ == '__main__':
    ### config list
    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    resetseed(2021)

    # fulllabel
    base_path = '/want/to/path'
    label_path = '/want/to/path'

    classname = ('class1','class2','class3','class4','5')
    batchsize=300
    lr = 1e-5
    epochs = 100
    gpu = 1
    save_name = './model/base'
    modelname = 'resnet18'
    
    # 
    checkpoint_callback = ModelCheckpoint(
                        monitor='valid_loss',
                        dirpath=save_name,
                        filename='model-{epoch:03d}-{valid_loss:.4f}',
                        save_top_k=3,
                        mode='min')

    ealrystopping = EarlyStopping(monitor='valid_loss',
                                min_delta=0.03, 
                                patience = 10, 
                                verbose=False,
                                mode='min')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    # dataset module

    # dataset inferencedataload
    wsin, roin = 1,0
    her2data = inferdataloderpl(base_path,label_path,wsin,roin,batchsize)
    
    # define her2model 

    last_savename = glob(f'{save_name}/*')[-1]
    print(last_savename)
    model = classify.load_from_checkpoint(checkpoint_path=last_savename,model=resnet(5,True,modelname),lr=lr)

    # set tensorboard 
    tb_logger = TensorBoardLogger('./logs',name=save_name)
    # set a trainer 
    
    trainer = pl.Trainer(
                        max_epochs=epochs, 
                        gpus = gpu,
                        # default_root_dir=save_name,
                        # checkpoint_callback =checkpoint_callback,
                        logger = tb_logger,
                        callbacks=[lr_monitor,checkpoint_callback])
    
    # trainer.fit(model,her2data)
    # train
    trainer.fit(model,her2data)
    useopenslide = True

    her2data.setup(stage='test',useopenslide=useopenslide)
    trainer.test(model,datamodule=her2data)

    result = np.stack(np.array(model.val_collector),axis=1)
    
    score =list(itertools.chain(*result[0]))
    resultpath =list(itertools.chain(*result[1]))

