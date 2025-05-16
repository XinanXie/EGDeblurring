from Dataset_s1s3 import get_train_dataset_GOPRO
from deblur_model import Event_Interface, WLNet

import torch
import torchvision
from tqdm import tqdm
from datetime import datetime
import os
import numpy as np
import lpips



SAVED_DIR=''
save_path_train=os.path.join(SAVED_DIR,'saved_train')
LOG_DIR=os.path.join(SAVED_DIR,'log')

def main():

    os.makedirs(SAVED_DIR, exist_ok=True)
    os.makedirs(save_path_train, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = get_device(3)

    patch_size=(240,320)
    # patch_size=(256,256)
    print('patch_size',patch_size)
    batch_size = 4
    num_epochs = 601

    net = WLNet(num_input_channel=3,num_feature_channel=256,num_event_channel=6).cuda(device)
    interface = Event_Interface(num_input_channel=6,num_feature_channel=256).cuda(device)

    #data
    train_data_set=get_train_dataset_GOPRO()
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, shuffle=True,num_workers=1)

    train_loop(net,interface,train_loader,batch_size,device,num_epochs = num_epochs,patch_size = patch_size)



def train_loop(model,interface,train_loader,batch_size,device,num_epochs = 401,patch_size=(360,640),lr=2e-4):

    optimizer=torch.optim.Adam([{'params': model.parameters()},{'params': interface.parameters()}],lr=lr)

    Loss_l1_fn=torch.nn.L1Loss()
    Loss_lpips_fn = lpips.LPIPS(net='vgg').to(device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs,eta_min=1e-7)

    crop=torchvision.transforms.Compose([torchvision.transforms.RandomCrop(patch_size),torchvision.transforms.RandomHorizontalFlip(p=0.5),torchvision.transforms.RandomVerticalFlip(p=0.5)])



    train_L1=[]
    # train_lpips=[]

    for e in range(num_epochs):

        print("\ntranining progress: epoch {}".format(e + 1),'model_WLNet_interface_demo4:')
        

        epoch_L1=[]
        # epoch_lpips=[]


        for input_batch, gt_batch,event_batch,name in tqdm(train_loader):


            input_batch=input_batch.cuda(device)
            gt_batch=gt_batch.cuda(device)
            event_batch=event_batch.cuda(device)

            input_batch = norm(input_batch)
            gt_batch = norm(gt_batch)

            gt_input_event_cat=torch.cat((gt_batch,input_batch,event_batch),dim=1)
            gt_input_event_cat=crop(gt_input_event_cat)
            gt_batch=gt_input_event_cat[:,0:3]
            input_batch=gt_input_event_cat[:,3:6]
            event_batch=gt_input_event_cat[:,6:12]

            event_batch_re = interface(event_batch)
            pred = model(input_batch,event_batch_re)

            loss_L1 = Loss_l1_fn(pred,gt_batch)
            # loss_lpips = (Loss_lpips_fn(pred,gt_batch)).sum()*0.1

            loss = loss_L1 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_L1.append(loss_L1.item())
            # epoch_lpips.append(loss_lpips.item())


        train_L1.append(np.mean(epoch_L1))
        # train_lpips.append(np.mean(epoch_lpips))

        print('epoch_L1:',np.mean(epoch_L1))
        # print('epoch_lpips:',np.mean(epoch_lpips))


        if (e+1)%50==0 : #debug

            os.makedirs(os.path.join(SAVED_DIR), exist_ok=True)

            name = datetime.now().strftime("WLNet_%d-%m-%Y_%H-%M")
            name ='{0}_{1}epoch'.format(name,e+1)
            fullpath = os.path.join(SAVED_DIR, name)
            torch.save(model.state_dict(), fullpath)
            print(f"SAVED MODEL AS:\n"
                f"{name}\n"
                f"in: {SAVED_DIR}")


            name_interface = datetime.now().strftime("Interface_%d-%m-%Y_%H-%M")
            name_interface='{0}_{1}epoch'.format(name_interface,e+1)
            fullpath = os.path.join(SAVED_DIR, name_interface)
            torch.save(interface.state_dict(), fullpath)
            print(f"SAVED MODEL AS:\n"
                f"{name_interface}\n"
                f"in: {SAVED_DIR}")

        if (e+1)%10==0 : #debug

            data = np.array([train_L1]).T    #,train_lpips
            filename = datetime.now().strftime("model_2_3_%d-%m-%Y_%H-%M")
            filename = '{0}_{1}epoch.csv'.format(filename,(e+1))
            fullpath = os.path.join(LOG_DIR, filename)
            np.savetxt(fullpath, data, delimiter=',')
                       

        scheduler.step()
       



def get_device(gpu=0,use_gpu=True):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    return device
            
            

def norm(temp):
    temp = (temp - 0.0) / (255.0 - 0.0)
    temp.clamp_(0.0, 1.0)

    return temp




if __name__=="__main__":
    main()