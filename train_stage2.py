import torch
import data as Data
import model as Model
import argparse
import core.logger as Logger
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from deblur_model import Event_Interface
import random
from model.sr3_modules import transformer
from Dataset_s1s3 import get_train_dataset_GOPRO
import torchvision.transforms as transforms
from tqdm import tqdm
import torchvision
# os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"



def norm(temp):
    temp = (temp - temp.min()) / (temp.max() - temp.min())
    temp = temp/0.5 - 1
    temp.clamp_(-1.0, 1.0)

    return temp


def norm_to01(temp):
    temp = (temp - temp.min()) / (temp.max() - temp.min())
    # temp = temp/0.5 - 1
    temp.clamp_(0.0, 1.0)

    return temp



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/deblur_1.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
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
    # Logger.setup_logger(None, opt['path']['log'],
    #                     'train', level=logging.INFO, screen=True)
    # Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    # logger = logging.getLogger('base')
    # logger.info(Logger.dict2str(opt))
    # tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # # Initialize WandbLogger
    # if opt['enable_wandb']:
    #     wandb_logger = WandbLogger(opt)
    #     wandb.define_metric('validation/val_step')
    #     wandb.define_metric('epoch')
    #     wandb.define_metric("validation/*", step_metric="val_step")
    #     val_step = 0
    # else:
    #     wandb_logger = None

    # ######### Set Seeds ###########
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    # Intrinsic dataset
    Intrinsic_dataset = get_train_dataset_GOPRO()
    Intrinsic_dataloader = torch.utils.data.DataLoader(Intrinsic_dataset,batch_size=opt["datasets"]["train"]["batch_size"],shuffle=True,num_workers=1,pin_memory=True)
    # logger.info('Initial Dataset Finished')
    diffusion = Model.create_model(opt)

    #interface
    device = torch.device('cuda')
    interface = Event_Interface(num_input_channel=6,num_feature_channel=256).cuda()
    model_path=''      
    interface.load_state_dict(torch.load(model_path,map_location=device))


    # model
    # logger.info('Initial Model Finished')
    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    # if opt['path']['resume_state']:
    #     logger.info('Resuming training from epoch: {}, iter: {}.'.format(
    #         current_epoch, current_step))
    to_pil = transforms.ToPILImage()
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    #opt['phase'] = 'test'
    patch_size=(240,320)
    crop=torchvision.transforms.Compose([torchvision.transforms.RandomCrop(patch_size),torchvision.transforms.RandomHorizontalFlip(p=0.5),torchvision.transforms.RandomVerticalFlip(p=0.5)])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            print('epoch:',current_epoch)
            for _, (input_batch, gt_batch,event_batch,name) in enumerate(tqdm(Intrinsic_dataloader, ncols=100)):
                current_step += 1
                if current_step > n_iter:
                    break

                input_batch_dif = norm(input_batch) 
                gt_batch_dif = norm(gt_batch)

                gt_input_event_cat=torch.cat((gt_batch_dif,input_batch_dif,event_batch),dim=1)
                gt_input_event_cat=crop(gt_input_event_cat)
                gt_batch_dif=gt_input_event_cat[:,0:3]
                input_batch_dif=gt_input_event_cat[:,3:6]
                event_batch=gt_input_event_cat[:,6:12]

                scale_factor = 4
                resize_size =(patch_size[0]//scale_factor,patch_size[1]//scale_factor)
                reszie=torchvision.transforms.Resize(resize_size)
                reszie_1=torchvision.transforms.Resize(patch_size)
                frame_blur_LR = reszie(input_batch_dif)
                frame_blur_SR = reszie_1(frame_blur_LR)

                event_batch_re = interface(event_batch.cuda())

                train_data = {'HR': gt_batch_dif, 'BL': input_batch_dif, 'event': event_batch,'event_feature':event_batch_re.detach(),'name': name, 'SR':frame_blur_SR}

                diffusion.feed_data(train_data)
                loss = diffusion.optimize_parameters()

                if current_step % opt['train']['print_freq'] ==0:
                    print("onlyEvent:epoch %d "%(current_epoch))
                    print("current loss is %f \n"%(loss))
                    
                    
                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    diffusion.save_network(current_epoch, current_step)
    if opt['phase'] == 'val':
        pass


