
import torch
import model as Model
import argparse
import logging
import core.logger as Logger
from core.wandb_logger import WandbLogger
# from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import wandb
from deblur_model import Event_Interface, WLNet
from sr_rj import event_filter
import utils
import random
from model.sr3_modules import transformer

import torchvision

from Dataset_s1s3 import get_test_dataset_GOPRO, get_train_dataset_GOPRO
from torch.utils import data

from tqdm import tqdm
from datetime import datetime


def norm_to01(temp):
    temp = (temp - temp.min()) / (temp.max() - temp.min())
    # temp = temp/0.5 - 1
    temp.clamp_(0.0, 1.0)

    return temp

def norm_255(temp):
    temp = (temp - 0.0) / (255.0 - 0.0)
    temp.clamp_(0.0, 1.0)

    return temp


def norm(temp):
    temp = (temp - temp.min()) / (temp.max() - temp.min())
    temp = temp/0.5 - 1
    temp.clamp_(-1.0, 1.0)

    return temp





if __name__ == "__main__":

    SAVED_DIR=''
    save_path_train=os.path.join(SAVED_DIR,'saved_train')
    LOG_DIR=os.path.join(SAVED_DIR,'log')
    os.makedirs(SAVED_DIR, exist_ok=True)
    os.makedirs(save_path_train, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/deblur_s3.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train') ##########
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # torch.cuda.set_device(opt['gpu_ids'][0])

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    # tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # ######### Set Seeds ###########
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    
    # model
    diffusion = Model.create_model(opt)

    device = torch.device('cuda')
    deblurnet = WLNet(num_input_channel=3,num_feature_channel=256,num_event_channel=6).cuda()
    model_path=''      
    deblurnet.load_state_dict(torch.load(model_path,map_location=device))
    deblurnet.train()

    interface = Event_Interface(num_input_channel=6,num_feature_channel=256).cuda(device)
    model_path=''  
    interface.load_state_dict(torch.load(model_path,map_location=device))
    interface.eval()

    lr=2e-4
    num_epochs = 601
    patch_size= (240,320)  #(256,256)
    batch_size = 1
    optimizer=torch.optim.Adam([{'params': deblurnet.parameters()}],lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs,eta_min=1e-7)


    train_data_set=get_train_dataset_GOPRO()
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, shuffle=True,num_workers=1)

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])


    Loss_l1_fn=torch.nn.L1Loss()
    crop=torchvision.transforms.Compose([torchvision.transforms.RandomCrop(patch_size),torchvision.transforms.RandomHorizontalFlip(p=0.5),torchvision.transforms.RandomVerticalFlip(p=0.5)])

    train_L1=[]
    train_letant=[]

    for e in range(num_epochs):

        print("\ntranining progress: epoch {}".format(e + 1))
        

        epoch_L1=[]
        epoch_letant=[]


        for input_batch, gt_batch,event_batch,name in tqdm(train_loader, ncols=100):


            input_batch=input_batch.cuda()
            gt_batch=gt_batch.cuda()
            event_batch=event_batch.cuda()

            input_01 = norm_255(input_batch)
            input_dif = norm(input_batch)
            gt_01 = norm_255(gt_batch)

            gt_input_event_cat=torch.cat((gt_01,input_01,input_dif,event_batch),dim=1)
            gt_input_event_cat=crop(gt_input_event_cat)
            gt_01=gt_input_event_cat[:,0:3]
            input_01=gt_input_event_cat[:,3:6]
            input_dif=gt_input_event_cat[:,6:9]
            event_batch=gt_input_event_cat[:,9:15]

            # 构造数据集
            scale_factor = 4
            resize_size =(patch_size[0]//scale_factor,patch_size[1]//scale_factor)
            reszie=torchvision.transforms.Resize(resize_size)#Cnan ,interpolation=Image.BICUBIC
            reszie_1=torchvision.transforms.Resize(patch_size)#Cnan
            input_dif_LR = reszie(input_dif)
            input_dif_SR = reszie_1(input_dif_LR)

            event_batch_re = interface(event_batch) 

            
            difference = input_dif - input_dif_SR
            input_cat = torch.cat([input_dif, difference],dim=1)
            out_event_feature = diffusion.netG.p_sample_loop_complete(input_cat.cuda(),event_batch.cuda())

            pred = deblurnet(input_01,out_event_feature.cuda())

            loss_L1 = Loss_l1_fn(pred,gt_01)
            loss_latent = Loss_l1_fn(out_event_feature.cuda(),event_batch_re)
            # loss_lpips = (Loss_lpips_fn(pred,gt_batch)).sum()*0.1

            loss = loss_L1 + loss_latent*0.01

            optimizer.zero_grad()
            diffusion.optG.zero_grad()
            loss.backward()
            optimizer.step()
            diffusion.optG.step()


            epoch_L1.append(loss_L1.item())
            epoch_letant.append(loss_latent.item())



        train_L1.append(np.mean(epoch_L1))
        train_letant.append(np.mean(epoch_letant))
        # train_lpips.append(np.mean(epoch_lpips))

        print('epoch_L1:',np.mean(epoch_L1))
        print('epoch_letant:',np.mean(epoch_letant))
        # print('epoch_lpips:',np.mean(epoch_lpips))


        if (e+1)%50==0 : #debug

            os.makedirs(os.path.join(SAVED_DIR), exist_ok=True)
            current_epoch = e+1
            current_step = (e+1)*515

            name = datetime.now().strftime("WLNet_%d-%m-%Y_%H-%M")
            name ='{0}_{1}epoch'.format(name,current_epoch)
            fullpath = os.path.join(SAVED_DIR, name)
            torch.save(deblurnet.state_dict(), fullpath)
            print(f"SAVED MODEL AS:\n"
                f"{name}\n"
                f"in: {SAVED_DIR}")

            
            diffusion.save_network(current_epoch, current_step)

        if (e+1)%10==0 : #debug

            data = np.array([train_L1,train_letant]).T    #,train_lpips
            filename = datetime.now().strftime("model_2_3_%d-%m-%Y_%H-%M")
            filename = '{0}_{1}epoch.csv'.format(filename,(e+1))
            fullpath = os.path.join(LOG_DIR, filename)
            np.savetxt(fullpath, data, delimiter=',')
                       

        scheduler.step()

