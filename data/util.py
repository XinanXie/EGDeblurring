import os
import torch
import torchvision
import random
import numpy as np

import cv2

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img * (min_max[1] - min_max[0]) + min_max[0]
    return img


def random_crop(stacked_img, patch_size):
    # img_size: int
    # stacked_image shape: 2*C*H*W type: tensor
    h, w = stacked_img.shape[2], stacked_img.shape[3]
    start_h = np.random.randint(low=0, high=(h - patch_size) + 1) if h > patch_size else 0
    start_w = np.random.randint(low=0, high=(h - patch_size) + 1) if w > patch_size else 0
    return stacked_img[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size]


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
# totensor = torchvision.transforms.ToTensor()
# hflip = torchvision.transforms.RandomHorizontalFlip()
# preresize = torchvision.transforms.Resize([256, 256])
crop=torchvision.transforms.Compose([torchvision.transforms.RandomCrop(256),torchvision.transforms.RandomHorizontalFlip(p=0.5),torchvision.transforms.RandomVerticalFlip(p=0.5)])

# def transform_augment(img_list, split='val', min_max=(0, 1), patch_size=160):
#     imgs = [preresize(img) for img in img_list]
#     imgs = [totensor(img) for img in imgs]
#     img_mask = imgs[-1]
#     img_mask = img_mask.repeat(3, 1, 1)
#     imgs[-1] = img_mask
#     imgs = torch.stack(imgs, 0)
#     if split == 'train':
#         imgs = random_crop(imgs, patch_size=patch_size)
#         imgs = hflip(imgs)
#     crop_h, crop_w = imgs.shape[2] % 16, imgs.shape[3] % 16
#     imgs = imgs[:, :, :imgs.shape[2] - crop_h, :imgs.shape[3] - crop_w]
#     imgs = torch.unbind(imgs, dim=0)

#     ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs[0:-1]]
#     ret_img.append(imgs[-1])
#     ret_img[-1] = torch.mean(ret_img[-1], 0, keepdim=True)
#     return ret_img

def norm_to01(temp):
    temp = (temp - temp.min()) / (temp.max() - temp.min())
    temp.clamp_(0.0, 1.0)

    return temp



def norm(temp):
    temp = (temp - temp.min()) / (temp.max() - temp.min())
    temp = temp/0.5 - 1
    temp.clamp_(-1.0, 1.0)

    return temp

# def eventnorm(tensor):
#     tensor_pos = tensor.clone()
#     tensor_pos[tensor_pos <= 0] = 0
#     if tensor_pos.max() != 0:
#         tensor_pos = tensor_pos / tensor_pos.max()

#     # 对小于0的元素进行归一化到 (-1, 0)
#     tensor_neg = tensor.clone()
#     tensor_neg[tensor_neg >= 0] = 0
#     if tensor_neg.min() != 0:
#         tensor_neg = tensor_neg / torch.abs(tensor_neg.min())

#     # 合并大于0和小于0的归一化结果
#     normalized_tensor = tensor_pos + tensor_neg

#     return normalized_tensor



#分通道归一化
def eventnorm(tensor_batch):

    C,_,_ = tensor_batch.shape
    tensor_out = torch.zeros_like(tensor_batch)

    for i in range(C):
        tensor = tensor_batch[i].unsqueeze(0)
        tensor_pos = tensor.clone()
        tensor_pos[tensor_pos <= 0] = 0
        if tensor_pos.max() != 0:
            tensor_pos = tensor_pos / tensor_pos.max()

        # 对小于0的元素进行归一化到 (-1, 0)
        tensor_neg = tensor.clone()
        tensor_neg[tensor_neg >= 0] = 0
        if tensor_neg.min() != 0:
            tensor_neg = tensor_neg / torch.abs(tensor_neg.min())

        # 合并大于0和小于0的归一化结果
        normalized_tensor = tensor_pos + tensor_neg
        tensor_out[i] = normalized_tensor


    return tensor_out

#for event input
def my_transform_augment(img_list, is_train):
    # imgs = [preresize(img) for img in img_list]
    # imgs = [totensor(img) for img in imgs]
    # img_event = imgs[-1]
    # img_event = img_event.repeat(3, 1, 1)
    # imgs[-1] = img_event
    # imgs = torch.stack(img_list, 0)
    
    if is_train == True:
        imgs=torch.cat(img_list,dim=0)
        imgs=crop(imgs)
        gt=norm(imgs[0:3])
        input=norm(imgs[3:6])
    else:
        gt = norm(img_list[0])
        input = norm(img_list[1])

    img_list = [gt,input]
    # crop_h, crop_w = imgs.shape[2] % 16, imgs.shape[3] % 16
    # imgs = imgs[:, :, :imgs.shape[2] - crop_h, :imgs.shape[3] - crop_w]
    # imgs = torch.unbind(imgs, dim=0)

    # ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs[0:-1]]
    # ret_img.append(imgs[-1])
    # ret_img[-1] = torch.mean(ret_img[-1], 0, keepdim=True)
    return img_list


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