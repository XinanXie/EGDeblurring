from time import time
import torch
from demo_param import count_parameters
import model as Model
import argparse
import logging
import core.logger as Logger
from metrics import SSIM,PSNR
import lpips
from core.wandb_logger import WandbLogger
# from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import wandb
from deblur_model import WLNet
from sr_rj import event_filter
import utils
import random
from model.sr3_modules import transformer

import torchvision
import cv2

from Dataset_s1s3 import get_test_dataset_GOPRO, get_train_dataset_GOPRO
from torch.utils import data

from tqdm import tqdm

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


def make_event_preview(events, mode='red-blue', num_bins_to_show=-1):
    # events: [1 x C x H x W] event tensor
    # mode: 'red-blue' or 'grayscale'
    # num_bins_to_show: number of bins of the voxel grid to show. -1 means show all bins.
    assert(mode in ['red-blue', 'grayscale'])
    if num_bins_to_show < 0:
        sum_events = torch.sum(events[0, :, :, :], dim=0).detach().cpu().numpy()
    else:
        sum_events = torch.sum(events[0, -num_bins_to_show:, :, :], dim=0).detach().cpu().numpy()

    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.zeros((sum_events.shape[0], sum_events.shape[1], 3), dtype=np.uint8)
        b = event_preview[:, :, 0]
        r = event_preview[:, :, 2]
        b[sum_events > 0] = 255
        r[sum_events < 0] = 255
    else:
        # Grayscale mode
        # normalize event image to [0, 255] for display
        m, M = -10.0, 10.0
        event_preview = np.clip((255.0 * (sum_events - m) / (M - m)).astype(np.uint8), 0, 255)

    return event_preview

if __name__ == "__main__":
    calculate_mae=torch.nn.L1Loss()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/deblur_test.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='val')
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

    # dataset
    # for phase, dataset_opt in opt['datasets'].items():
    #     if phase == 'train' and args.phase != 'val':
    #         train_set = Data.create_dataset_GoPro_train(dataset_opt)
    #         train_loader = Data.create_dataloader(
    #             train_set, dataset_opt, phase)
    #     elif phase == 'val':
    #         val_set = Data.create_dataset_GoPro_val(dataset_opt)
    #         val_loader = Data.create_dataloader(
    #             val_set, dataset_opt, phase)

    batch_size = 1
    test_data_set=get_test_dataset_GOPRO()
    val_loader = data.DataLoader(test_data_set, batch_size=batch_size, shuffle=False,num_workers=1)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    total_params = count_parameters(diffusion.netG.denoise_fn)
    print(f"Total number of parameters: {total_params}")

    device = torch.device('cuda')
    deblurnet = WLNet(num_input_channel=3,num_feature_channel=256,num_event_channel=6).cuda()
    model_path=''     
    deblurnet.load_state_dict(torch.load(model_path,map_location=device))

    # degradation predict model
    # model_restoration = transformer.Uformer()
    # model_restoration.cuda()
    # path_chk_rest_student = '../models/model_epoch_450.pth'
    # utils.load_checkpoint(model_restoration, path_chk_rest_student)
    # model_restoration.eval()

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])


    ssim_fn=SSIM()
    psnr_fn=PSNR(max_val=1.0)
    lpips_fn = lpips.LPIPS(net='vgg').cuda()

    ssim_list = []  # loss of each batch
    lpips_list = []  # loss of each batch
    psnr_list = []  # loss of each batch



    if opt['phase'] == 'train':
        pass
    else:
        with torch.no_grad():
            logger.info('Begin Model Evaluation.')
            avg_psnr = 0.0
            avg_ssim = 0.0
            avg_mae = 0.0
            idx = 0
            result_path = '{}'.format(opt['path']['results'])
            os.makedirs(result_path, exist_ok=True)
            for _,  (input_batch, gt_batch,event_batch,name) in enumerate(tqdm(val_loader)):
                idx += 1
                # diffusion.feed_data(val_data)
                # sr_img = diffusion.netG.p_sample_loop(val_data['SR'].cuda(),val_data['event'].cuda())
                # visuals = diffusion.my_get_current_visuals()

                input_org=input_batch.cuda()
                gt_org=gt_batch.cuda()
                event_batch=event_batch.cuda()

                input_dif = norm(input_org)
                input_01 = norm_255(input_org)
                gt_01 = norm_255(gt_org)

                image_size = (720,1280)
                scale_factor = 4
                resize_size =(image_size[0]//scale_factor,image_size[1]//scale_factor)
                reszie=torchvision.transforms.Resize(resize_size)
                reszie_1=torchvision.transforms.Resize(image_size)
                input_dif_LR = reszie(input_dif)
                input_dif_SR = reszie_1(input_dif_LR)

                difference = input_dif - input_dif_SR
                input_cat = torch.cat([input_dif, difference],dim=1)
                out_event_feature = diffusion.netG.p_sample_loop(input_cat.cuda(),event_batch.cuda())

                pred = deblurnet(input_01.cuda(),out_event_feature.cuda())

                ssim_test=ssim_fn(pred,gt_01).sum()
                lpips_test=lpips_fn(pred,gt_01).sum()
                psnr_test=psnr_fn(pred,gt_01).sum()

                ssim_list.append(ssim_test.item())
                lpips_list.append(lpips_test.item())
                psnr_list.append(psnr_test.item())


                path_save = os.path.join(result_path,'{0}_{1}_predict.png'.format(name[0][0],name[1][0].item()))
                torchvision.utils.save_image(pred,path_save,normalize=False)
                

        ssim_mean=np.mean(ssim_list)
        lpips_mean=np.mean(lpips_list)
        psnr_mean=np.mean(psnr_list)


        path_val=os.path.join(result_path,'0.txt')

        with open(path_val,'w') as f:
            f.write(f'ssim: {ssim_mean} ')
            f.write(f'lpips: {lpips_mean} ')
            f.write(f'psnr: {psnr_mean} ')
            
             
        print(f'ssim: {ssim_mean} ')
        print(f'lpips: {lpips_mean} ')
        print(f'psnr: {psnr_mean} ')

       