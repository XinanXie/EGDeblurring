from torch.utils import data as data
from torch import from_numpy
import torchvision
import torch
import pandas as pd
import os
import cv2
import h5py
# import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from PIL import Image as Image
import data.util as Util
from torch.utils.data import ConcatDataset

# from utils import make_event_preview


class H5ImageDataset_deblurSR(data.Dataset):

    def __init__(self, data_path, name, is_train,image_size=(704,1280),crop_size=256,scale_factor=4,return_voxel=True, return_frame=True, return_gt_frame=True,
            return_mask=False, norm_voxel=True):

        super(H5ImageDataset_deblurSR,self).__init__()

        self.data_path=data_path
        self.return_voxel=return_voxel
        self.return_frame=return_frame
        self.return_gt_frame=return_gt_frame
        self.return_mask=return_mask

        self.norm_voxel=norm_voxel
        self.is_train = is_train

        self.h5_file=None

        self.name = name
    
        scale_factor = 4
        
        if is_train is True:
            resize_size =(crop_size//scale_factor,crop_size//scale_factor)
            self.reszie=torchvision.transforms.Resize(resize_size)#Cnan ,interpolation=Image.BICUBIC
            self.reszie_1=torchvision.transforms.Resize(crop_size)#Cnan
        else:
            resize_size = (image_size[0]//scale_factor,image_size[1]//scale_factor)
            self.reszie=torchvision.transforms.Resize(resize_size)#Cnan ,interpolation=Image.BICUBIC
            self.reszie_1=torchvision.transforms.Resize(image_size)#Cnan

        

        self.crop=torchvision.transforms.Compose([torchvision.transforms.RandomCrop(crop_size),torchvision.transforms.RandomHorizontalFlip(p=0.5),torchvision.transforms.RandomVerticalFlip(p=0.5)])

        

        with h5py.File(self.data_path, 'r') as file:
            self.dataset_len = len(file['images'].keys())


    def get_frame(self, index):
        """
        Get frame at index
        @param index The index of the frame to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['images']['image{:09d}'.format(index)][:]


    def get_gt_frame(self, index):
        """
        Get gt frame at index
        @param index: The index of the gt frame to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['sharp_images']['image{:09d}'.format(index)][:]

    def get_voxel(self, index):
        """
        Get voxels at index
        @param index The index of the voxels to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['voxels']['voxel{:09d}'.format(index)][:]  

    def get_mask(self, index):
        """
        Get event mask at index
        @param index The index of the event mask to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['masks']['mask{:09d}'.format(index)][:]

    def bgr2rgb(self,x):

        x_rgb=torch.zeros_like(x)
        if len(x.shape)==3:
            x_rgb[0,:,:]=x[-1,:,:]
            x_rgb[1,:,:]=x[1,:,:]
            x_rgb[-1,:,:]=x[0,:,:]

        if len(x.shape)==4:
            x_rgb[:,0,:,:]=x[:,-1,:,:]
            x_rgb[:,1,:,:]=x[:,1,:,:]
            x_rgb[:,-1,:,:]=x[:,0,:,:]

        return x_rgb

    # def __getitem__(self, index):

    #     if index < 0 or index >= self.__len__():
    #         raise IndexError

    #     frame_blurr=from_numpy(self.get_frame(index))
    #     frame_blurr=self.bgr2rgb(frame_blurr)
    #     frame_gt=from_numpy(self.get_gt_frame(index))
    #     frame_gt=self.bgr2rgb(frame_gt)
    #     events=from_numpy(self.get_voxel(index))

    #     name=(self.name,index)

    #     if self.is_train is True:
    #         imgs = torch.cat((frame_gt,frame_blurr,events),dim=0)
    #         imgs = self.crop(imgs)
    #         frame_gt=imgs[0:3]
    #         frame_blurr=imgs[3:6] 
    #         events=imgs[6:12]
    #     else:
    #         frame_gt=frame_gt[:,:704,:]
    #         frame_blurr=frame_blurr[:,:704,:]
    #         events=events[:,:704,:] 

    #     # events_blur_LR = self.reszie(events)
    #     # events_blur_SR = self.reszie_1(events_blur_LR)
        
    #     frame_gt = Util.norm(frame_gt)
    #     frame_blurr = Util.norm(frame_blurr)
    #     events = Util.eventnorm(events)

    #     frame_blur_LR = self.reszie(frame_blurr)
    #     frame_blur_SR = self.reszie_1(frame_blur_LR)

        

    #     if self.is_train is True:
    #         return {'HR': frame_gt, 'BL': frame_blurr, 'event': events, 'name': name, 'SR':frame_blur_SR} #'event': events_blur_SR, 'SR':frame_blur_SR, 'LR': frame_blur_LR
            
    #     else:
    #         return {'HR': frame_gt, 'BL': frame_blurr, 'event': events,  'name': name, 'SR':frame_blur_SR} #'event': events_blur_SR,'LR': Util.norm(frame_blur_LR), 'SR':frame_blur_SR, 'LR': frame_blur_LR,

    def __getitem__(self, index): 

        if index < 0 or index >= self.__len__():
            raise IndexError

        frame_blurr=from_numpy(self.get_frame(index))
        frame_blurr=self.bgr2rgb(frame_blurr)
        frame_gt=from_numpy(self.get_gt_frame(index))
        frame_gt=self.bgr2rgb(frame_gt)
        events=from_numpy(self.get_voxel(index))

        name=(self.name,index)

        frame_gt = Util.norm(frame_gt)
        frame_blurr = Util.norm(frame_blurr)
        # events = Util.eventnorm(events)

        if self.is_train is True:
            imgs = torch.cat((frame_gt,frame_blurr,events),dim=0)
            imgs = self.crop(imgs)
            frame_gt=imgs[0:3]
            frame_blurr=imgs[3:6] 
            events=imgs[6:12]
        else:
            frame_gt=frame_gt[:,:704,:]
            frame_blurr=frame_blurr[:,:704,:]
            events=events[:,:704,:] 

        # events_blur_LR = self.reszie(events)
        # events_blur_SR = self.reszie_1(events_blur_LR)
        
        
       

        frame_blur_LR = self.reszie(frame_blurr)
        frame_blur_SR = self.reszie_1(frame_blur_LR)

        

        if self.is_train is True:
            return {'HR': frame_gt, 'BL': frame_blurr, 'event': events, 'name': name, 'SR':frame_blur_SR} #'event': events_blur_SR, 'SR':frame_blur_SR, 'LR': frame_blur_LR
            
        else:
            return {'HR': frame_gt, 'BL': frame_blurr, 'event': events,  'name': name, 'SR':frame_blur_SR} #'event': events_blur_SR,'LR': Util.norm(frame_blur_LR), 'SR':frame_blur_SR, 'LR': frame_blur_LR,


    def __len__(self):
        return self.dataset_len




# def get_train_dataset_REBlur():
#     data_path='/home/xinan/Deblurr1/datasets/REBlur/train'
#     ListTrainData=os.listdir(data_path)
#     print(f'from {data_path} get {len(ListTrainData)} h5file as trianing dataset')
    
#     datasets = []

#     for name in ListTrainData:
#         path=os.path.join(data_path,name)
#         # print(f'subdataset:{name}')
#         dataset=H5ImageDataset(path,name)
#         datasets.append(dataset)

#     TrainDataSet=ConcatDataset(datasets)

#     return TrainDataSet


# def get_test_dataset_REBlur():
#     data_path='/home/xinan/Deblurr1/datasets/REBlur/test'
#     ListTestData=os.listdir(data_path)
#     print(f'from {data_path} get {len(ListTestData)} h5file as test dataset')
    
#     datasets = []
#     for name in ListTestData:
#         path=os.path.join(data_path,name)
#         # print(f'subdataset:{name}')
#         dataset=H5ImageDataset(path,name)
#         datasets.append(dataset)


#     TestDataSet=ConcatDataset(datasets)

#     return TestDataSet




def get_train_dataset_GOPRO():
    data_path='/home/xinan/Deblurr1/datasets/GOPRO/train'
    ListTrainData=os.listdir(data_path)
    print(f'from {data_path} get {len(ListTrainData)} h5file as trianing dataset')
    
    datasets = []

    for name in ListTrainData:
        path=os.path.join(data_path,name)
        # print(f'subdataset:{name}')
        dataset=H5ImageDataset_deblurSR(path,name,is_train=True)
        datasets.append(dataset)

    TrainDataSet=ConcatDataset(datasets)

    return TrainDataSet


def get_test_dataset_GOPRO():
    data_path='/home/xinan/Deblurr1/datasets/GOPRO/test'
    ListTestData=os.listdir(data_path)
    print(f'from {data_path} get {len(ListTestData)} h5file as test dataset')
    
    datasets = []
    for name in ListTestData:
        path=os.path.join(data_path,name)
        # print(f'subdataset:{name}')
        dataset=H5ImageDataset_deblurSR(path,name,is_train=False)
        datasets.append(dataset)


    TestDataSet=ConcatDataset(datasets)

    return TestDataSet





# class DeblurDataset(data.Dataset):
#     def __init__(self, image_dir = '/home/xinan/datasetGOPRO/test/test_GOPRO_crop_240x320/', transform=None, is_test=True):
#         self.image_dir = image_dir
#         self.image_list = os.listdir(os.path.join(image_dir, 'input'))
#         self._check_image(self.image_list)
#         self.image_list.sort()
#         self.transform = transform
#         self.is_test = is_test

#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         image = Image.open(os.path.join(self.image_dir, 'input', self.image_list[idx]))
#         label = Image.open(os.path.join(self.image_dir, 'target', self.image_list[idx]))

#         if self.transform:
#             image, label = self.transform(image, label)
#         else:
#             image = F.to_tensor(image)
#             label = F.to_tensor(label)
#         if self.is_test:
#             name = self.image_list[idx]
#             return image, label, name
#         return image, label

#     @staticmethod
#     def _check_image(lst):
#         for x in lst:
#             splits = x.split('.')
#             if splits[-1] not in ['png', 'jpg', 'jpeg']:
#                 raise ValueError




# if __name__=="__main__":

    

#     # train_data_set=get_train_dataset_GOPRO()
#     # train_loader = data.DataLoader(train_data_set, batch_size=1, shuffle=False)
#     print('start!')
#     data_set=get_test_dataset_REBlur()
#     loader = data.DataLoader(data_set, batch_size=1, shuffle=True)

#     data_set_d=get_train_dataset_REBlur()
#     loader_d = data.DataLoader(data_set_d, batch_size=1, shuffle=True)

#     for x,y,e,name in loader:
        
#         print('x shape:',x.shape)
#         print('y shape:',y.shape)
#         print('e shape:',e.shape)
#         print('name:',name)

#         x=x*1.0
#         y=y*1.0

#         torchvision.utils.save_image(x,'result/x.png',normalize=True)
#         torchvision.utils.save_image(y,'result/y.png',normalize=True)
#         pre=make_event_preview(e)
#         cv2.imwrite('result/e.png',pre)


#         print('data_set_d.__len__():',data_set_d.__len__())
#         unpair_index=torch.randint(0,data_set_d.__len__(),(1,))
#         _,unpair_sharp,_,unpair_name = data_set_d.__getitem__(unpair_index.item())
#         print('unpair_name:',unpair_name)

#         torchvision.utils.save_image(unpair_sharp*1.0,'result/unpair_sharp.png',normalize=True)



#         break

