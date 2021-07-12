import random , os , itertools, yaml,argparse
from pandas.io.sql import read_sql_query
import numpy as np
from glob import glob
from natsort import natsorted

# torch code
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchsummary

from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics.functional import accuracy

# my code
from HER2dataloader import HER2dataloderpl , inferHER2dataloderpl
from utils import *
from models import choosemodel
from pytorch_grad_cam import GradCAM,ScoreCAM
import cv2

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0), 
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class HER2classify(pl.LightningModule): 
    def __init__(self,model,lr,opt): 
        super().__init__()
        self.save_hyperparameters(opt)
        self.opt = opt
        self.encoder = model
        self.lr = lr
        self.val_collector, self.train_collector = [] , []
        # self.hparams = opt

        # self.weight = torch.tensor([0.9, 0.76740331, 0.8       , 0.92320442, 0.93314917],dtype=torch.float32).to('cuda:0')
        self.weight = torch.tensor([1,1,1,1,1],dtype=torch.float32).to('cuda:0')
        
        #### add grad cam 
        print(" ####Use Grad cam!!!!###")
        target_layer = model.layer4[-1]
        print(target_layer)
        self.cam = GradCAM(model=model,target_layer=target_layer,use_cuda=True)
        self.cam.batchsize = self.opt['batchsize']

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
        y = list(itertools.chain(*collector[:,0]))
        preds = list(itertools.chain(*collector[:,1]))
        print(len(y),len(preds))
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
        x,y,_  = batch 
        pred = self.encoder(x)
        loss = F.cross_entropy(pred,y,weight=self.weight)
        
        if evalu == True: 
            self.evaluate(pred,y,phase)
        
        return loss
    
    
    def training_step(self,batch, batch_idx): 
        phase = 'train'
        loss =self.shared_step(batch,phase,True)
        self.log(f'{phase}_loss',loss, prog_bar=True, logger=True)
        
        al_batch = batch[0:2]+[self.preds]
        _,y,preds = list(map(lambda x: x.detach().cpu().numpy(),al_batch))
        self.train_collector.append([y,preds])
        
        return loss 

    def trainion_epoch_end(self,train_step_outputs): 
        # confusion matric
        self.plot_confusionmatric(self.train_collector,'train',True)
    
    def validation_step(self,batch, batch_idx): 
        phase='valid'

        loss = self.shared_step(batch,phase,True)   
        self.log(f'{phase}_loss',loss, prog_bar=True, logger=True)
        
        al_batch = batch[0:2]+[self.preds]
        x,y,preds = list(map(lambda x: x.detach().cpu().numpy(),al_batch))
        
        self.plot_gridimg(x,y,preds,5,phase)
        self.val_collector.append([y,preds])
        
        return loss 
    
    def validation_epoch_end(self,validation_step_outputs): 
        # confusion matric
        self.plot_confusionmatric(self.val_collector,'valid',True)
        
    def test_step(self,batch, batch_idx): 

        phase='test'
        self.inimg,self.orimg = batch[0],batch[2]
        preds = torch.argmax(self.forward(batch),dim=1).detach().cpu().numpy()

        # self.log(f'{phase}_loss',loss, prog_bar=True, logger=True)
        # x,y,preds = list(map(lambda x: x.detach().cpu().numpy(),al_batch))
        # self.plot_gridimg(x,y,preds,5,phase)
        
        self.val_collector.append([preds,batch[1]])
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--yamldir',default='./config.yaml', type=str)
    args = parser.parse_args()
    # load yaml config file
    
    with open(args.yamldir) as f: 
        opt = yaml.load(f, Loader=yaml.FullLoader)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['opt_gpus']
    resetseed(opt['randomseed'])

    # set a callback function 
    ####
    checkpoint_callback = ModelCheckpoint(
                        monitor='valid_loss',
                        dirpath=opt['checkpoint_path'],
                        filename='PathQuant_her2-{epoch:03d}-{valid_loss:.4f}',
                        save_top_k=3,
                        mode='min')

    ealrystopping = EarlyStopping(monitor='valid_loss',
                                min_delta=0.03, 
                                patience = 10, 
                                verbose=False,
                                mode='min')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    ####

    # set tensorboard 
    tb_logger = TensorBoardLogger(opt['tensorboard_path'],name=opt['checkpoint_path'])
    
    # set a trainer 
    trainer = pl.Trainer(
                        max_epochs=opt['epochs'], 
                        gpus = opt['gpus'],
                        default_root_dir=opt['checkpoint_path'],
                        logger = tb_logger,
                        check_val_every_n_epoch=1,
                        callbacks=[lr_monitor,ealrystopping,checkpoint_callback])
    
    # select model 
    mymodel = choosemodel(5,True,opt['modelname'])
    # dataset module
    if opt['Training'] == True:
        her2data = HER2dataloderpl(opt['base_path'], opt['label_path'],opt['batchsize'])
        model = HER2classify(mymodel,opt['lr'],opt)

        torchsummary.summary(model,(1,3,256,256),device='cpu')
        # torchsummary()
        trainer.fit(model,her2data)
    
    # dataset inferencedataload
    else: 
        her2data = inferHER2dataloderpl(opt['base_path'], opt['base_path'],opt['wsin'],opt['roin'],opt['batchsize'])
        checkpath = opt['checkpoint_path']
        last_savename = natsorted(glob(f'{checkpath}/*.ckpt'))[-1] # import last checkpoint
        model = HER2classify.load_from_checkpoint(checkpoint_path=last_savename,model=mymodel,lr=opt['lr'],opt=opt)

        her2data.setup(stage='test',useopenslide=opt['useopenslide'])
        
        # n = 80
        # sample,orimg = next(iter(her2data.test_dataloader()))[0],next(iter(her2data.test_dataloader()))[2]
        
        ### grad cam ####
        ####################################################################################
        # sample = sample[n:n+1]
        # orimg = orimg[n]
        # print(sample.shape,'123123')
        # mymodel.eval().cuda()
        # target_layer = mymodel.layer4[-1]
        # cam = GradCAM(model=model,target_layer=target_layer,use_cuda=True)
        # cam.batchsize = opt['batchsize']

        # cv2.imwrite('origin.png',(orimg.numpy()*255).astype(np.uint8)[:,:,::-1])
        # grayscale_cam = cam(input_tensor=sample.unsqueeze(0),target_category=0,)
        # cam_image=  show_cam_on_image(orimg.numpy()[:,:,::-1],grayscale_cam[0,:],use_rgb=False)
        # cv2.imwrite('camsample1.png',cam_image)
        # grayscale_cam = cam(input_tensor=sample.unsqueeze(0),target_category=1)
        # cam_image=  show_cam_on_image(orimg.numpy()[:,:,::-1],grayscale_cam[0,:],use_rgb=False)
        # cv2.imwrite('camsample2.png',cam_image)
        # grayscale_cam = cam(input_tensor=sample.unsqueeze(0),target_category=2)
        # cam_image=  show_cam_on_image(orimg.numpy()[:,:,::-1],grayscale_cam[0,:],use_rgb=False)
        # cv2.imwrite('camsample3.png',cam_image)
        # grayscale_cam = cam(input_tensor=sample.unsqueeze(0),target_category=3)
        # cam_image=  show_cam_on_image(orimg.numpy()[:,:,::-1],grayscale_cam[0,:],use_rgb=False)
        # cv2.imwrite('camsample4.png',cam_image)
        # grayscale_cam = cam(input_tensor=sample.unsqueeze(0),target_category=4)
        # cam_image=  show_cam_on_image(orimg.numpy()[:,:,::-1],grayscale_cam[0,:],use_rgb=False)
        # cv2.imwrite('camsample5.png',cam_image)
        # grayscale_cam = cam(input_tensor=sample.unsqueeze(0),target_category=5)
        # cam_image=  show_cam_on_image(orimg.numpy(),grayscale_cam[n,:],use_rgb=False)
        # cv2.imwrite('camsample6.png',cam_image)
        ########################################################################################

        trainer.test(model,datamodule=her2data)
        
        if opt['WSIsave'] == True: 
            result = np.stack(np.array(model.val_collector),axis=1)        
            score =list(itertools.chain(*result[0]))
            resultpath =list(itertools.chain(*result[1]))

            #save classify result usimg pilot model 
            # basepathes = opt['base_path']
            # savecsv([resultpath,score],f'{basepathes}')

            if opt['useopenslide'] == True:
                location = np.array(resultpath)
                patchimages = her2data.images
                recover_wsi(patchimages,opt['label_path'],location,score,opt['checkpoint_path'])

            elif opt['useopenslide'] == False:
                make_wsi(resultpath,opt['label_path'],opt['wsin'],opt['roin'],score,opt['checkpoint_path'])

