from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models


# general libs
import cv2
# import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import random
import argparse
import glob

# constants
DAVIS_ROOT = '/home/chenjian/dataset/DAVIS_2017/DAVIS'
LSE_ROOT = '/root/dataset/seqsets'
PALETTE = [
  0, 0, 0,
  31, 119, 180,
  174, 199, 232,
  255, 127, 14,
  255, 187, 120,
  44, 160, 44,
  152, 223, 138,
  214, 39, 40,
  255, 152, 150,
  148, 103, 189,
  197, 176, 213,
  140, 86, 75,
  196, 156, 148,
  227, 119, 194,
  247, 182, 210,
  127, 127, 127,
  199, 199, 199,
  188, 189, 34,
  219, 219, 141,
  23, 190, 207,
  158, 218, 229
]


class font:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

    
# special print functions
def print_dbg(string):
    print('{}{}debug{} {}'.format(font.BOLD, font.RED, font.END, string))


'''
# 将labels转化为独热编码
例如：
labels = [2, 0, 1] , num_objects = 3
则转变后的labels为：
[[0, 0, 1], # 对应2
 [1, 0, 0], # 对应0
 [0, 1, 0]] # 对应1
最后再用ToCudaVariable,将生成的独热编码张量转移到GPU上
'''
def ToOneHot(labels, num_objects):
    print(labels)
    labels = labels.view(-1,1)
    labels = torch.eye(num_objects).index_select(dim=0, index=labels)
    return ToCudaVariable(labels)

'''
输入：二维numpy数组：[num_classes, H, W]
np.argmax返回沿指定轴的最大值索引，即概率最高的类别
输出的形状：[H, W], 每个像素点的值表示该点所属的类别
最后将输出转化为uint8格式
'''
def ToLabel(E):
    fgs = np.argmax(E, axis=0).astype(np.float32)
    return fgs.astype(np.uint8)

'''
输入应为一个包含多个tensor的列表
输出为一个包含Variable对象的列表，且其中所有张量都被转移到GPU上（cuda可用）
volatile用于指示一个张量是否应该是不可变的，再pytorch0.4.0及之后的版本中已被废弃，改用torch.no_grad()上下文管理器
<<<<<<< HEAD
使用较新版本的pytorch,应考虑改写函数如下:
=======
使用较新版本的pytorch,应考虑改写函数如下：
>>>>>>> a83a89ced627312f6f08551a53ec641f289c52b0
'''
def ToCudaVariable(xs):  
    if torch.cuda.is_available():  
        return [x.cuda() for x in xs]  
    else:  
        return xs

# def ToCudaVariable(xs, volatile=False):
#     if torch.cuda.is_available():
#         return [Variable(x.cuda(), volatile=volatile) for x in xs]
#     else:
#         return [Variable(x, volatile=volatile) for x in xs]

# 计算IOU
def iou(pred, gt):
    pred = pred.squeeze().cpu().data.numpy()
    pred = ToLabel(pred)
    gt = gt.squeeze().cpu().data.numpy()
    agg = pred + gt
    i = float(np.sum(agg == 2))
    u = float(np.sum(agg > 0))
    return i / u

'''
上采样函数，线性插值
输入：x[1, C, H, W] , size[H', W']
输出： [1, C, H', W']
'''
def upsample(x, size):
    x = x.numpy()[0]
    dsize = (size[1], size[0])
    x = cv2.resize(x, dsize=dsize, interpolation=cv2.INTER_LINEAR)
    return torch.unsqueeze(torch.from_numpy(x), dim=0)

'''
输入xs应该为一个张量列表
输出为缩放后的张量列表
'''
def downsample(xs, scale):
    if scale == 1:
        return xs

    # find new size dividable by 32
    h = xs[0].size()[2] 
    w = xs[0].size()[3]
    
    new_h = int(h * scale)
    new_w = int(w * scale) 
    new_h = new_h + 32 - new_h % 32
    new_w = new_w + 32 - new_w % 32

    dsize = (new_w, new_h)
    ys = []
    for x in xs:
        x = x.numpy()[0] # c,h,w
        if x.ndim == 3:
            x = np.transpose(x, [1,2,0])
            x = cv2.resize(x, dsize=dsize, interpolation=cv2.INTER_LINEAR)
            x = np.transpose(x, [2,0,1])
        else:
            x = cv2.resize(x, dsize=dsize, interpolation=cv2.INTER_LINEAR)

        ys.append(torch.unsqueeze(torch.from_numpy(x), dim=0))

    return ys


class DAVIS(data.Dataset):
    '''
    Dataset for DAVIS
    '''
    def __init__(self, root, imset='2016/val.txt', resolution='480p', multi_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = [] # 用于存放video名字的列表
        self.num_frames = {} # 用于存放每个video对应帧数的字典
        self.num_objects = {} # 存放对应video含有的对象个数
        self.shape = {} # 存放对应video第1帧图像大小
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)

        self.MO = multi_object

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        if self.MO:
            num_objects = self.num_objects[video]
        else:
            num_objects = 1
        info['num_objects'] = num_objects

        # raw_frames.shape[num_frames, H, W, 3], 值随机初始化， 数值类型float32
        raw_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        # raw_masks.shape[num_frames, H, W], 值随机初始化， 数值类型uint8
        raw_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
        for f in range(self.num_frames[video]):
            # 依次读取图片，并作归一化，再存入对应raw_frames中
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            raw_frames[f] = np.array(Image.open(img_file).convert('RGB'))/255.
            # 若当前帧存在对应掩码，则读取并转为调色板模式，若没有，则读入第一帧图像的掩码
            try:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  #allways return first frame mask
                raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            except:
                mask_file = os.path.join(self.mask_dir, video, '00000.png')
                raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)

            if self.MO:
                raw_masks[f] = raw_mask
            else:
                raw_masks[f] = (raw_mask != 0).astype(np.uint8)
        # 循环结束后，raw_frames为一个数组，包含一个视频序列所有帧归一化后的值
        # raw_masks，返回对应序列的掩码图像，调色板模式
            
        # make One-hot channel is object index
        # oh_masks.shape[num_frames, H, W, num_objects]
        oh_masks = np.zeros((self.num_frames[video],)+self.shape[video]+(num_objects,), dtype=np.uint8)
        for o in range(num_objects):
            oh_masks[:,:,:,o] = (raw_masks == (o+1)).astype(np.uint8)


        # padding size to be divide by 32
        nf, h, w, _ = oh_masks.shape
        new_h = h + 32 - h % 32
        new_w = w + 32 - w % 32
        # print(new_h, new_w)
        # 对oh_masks和raw_frames进行填充
        lh, uh = (new_h-h) / 2, (new_h-h) / 2 + (new_h-h) % 2
        lw, uw = (new_w-w) / 2, (new_w-w) / 2 + (new_w-w) % 2
        lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)
        pad_masks = np.pad(oh_masks, ((0,0),(lh,uh),(lw,uw),(0,0)), mode='constant')
        pad_frames = np.pad(raw_frames, ((0,0),(lh,uh),(lw,uw),(0,0)), mode='constant')
        info['pad'] = ((lh,uh),(lw,uw))

        th_frames = torch.unsqueeze(torch.from_numpy(np.transpose(pad_frames, (3, 0, 1, 2)).copy()).float(), 0)
        th_masks = torch.unsqueeze(torch.from_numpy(np.transpose(pad_masks, (3, 0, 1, 2)).copy()).long(), 0)
        # 输出的th_frames.shape[1, 3, num_frames, H, W],归一化的数组，每一个值在0-1之间，表示对应像素值大小
        # th_masks.shape[1, num_objects, num_framnes, H, W],one-hot编码后的数组，每一个值为0或1,表示对应的类别，应该从num_objects维度去看。
        return th_frames, th_masks, info
    
class LSE(data.Dataset):
    '''
    Dataset for DAVIS
    '''
    def __init__(self, root, split='train', multi_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'all/Annotations/')
        self.image_dir = os.path.join(root, 'all/JPEGImages/')
        # _imset_dir = os.path.join(root, '')
        _imset_f = os.path.join(root, split+'.txt')

        self.videos = [] # 用于存放video名字的列表
        self.num_frames = {} # 用于存放每个video对应帧数的字典
        self.num_objects = {} # 存放对应video含有的对象个数
        self.shape = {} # 存放对应video第1帧图像大小
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                list_mask = os.listdir(os.path.join(self.mask_dir, _video))
                mask_path = list_mask[0]
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, mask_path)).convert('L'))
                self.num_objects[_video] = 1
                self.shape[_video] = np.shape(_mask)

        self.MO = multi_object

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        if self.MO:
            num_objects = self.num_objects[video]
        else:
            num_objects = 1
        info['num_objects'] = num_objects

        # raw_frames.shape[num_frames, H, W, 3], 值随机初始化， 数值类型float32
        raw_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        # raw_masks.shape[num_frames, H, W], 值随机初始化， 数值类型uint8
        raw_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
        img_list = os.listdir(os.path.join(self.image_dir, video))
        img_list.sort(key=lambda x : int(x.split('.')[0]))
        mask_list = os.listdir(os.path.join(self.mask_dir, video))
        mask_list.sort(key=lambda x : int(x.split('.')[0]))

        for f in range(self.num_frames[video]):
            # 依次读取图片，并作归一化，再存入对应raw_frames中
            img_file = os.path.join(self.image_dir, video, img_list[f])
            raw_frames[f] = np.array(Image.open(img_file).convert('RGB'))/255.
            # 若当前帧存在对应掩码，若没有，则读入第一帧图像的掩码
            try:
                mask_file = os.path.join(self.mask_dir, video, mask_list[f])  #allways return first frame mask
                raw_mask = np.array(Image.open(mask_file).convert('L'), dtype=np.uint8)
            except:
                mask_file = os.path.join(self.mask_dir, video, mask_list[0])
                raw_mask = np.array(Image.open(mask_file).convert('L'), dtype=np.uint8)

            if self.MO:
                raw_masks[f] = raw_mask
            else:
                raw_masks[f] = (raw_mask != 0).astype(np.uint8)
        # 循环结束后，raw_frames为一个数组，包含一个视频序列所有帧归一化后的值
        # raw_masks，返回对应序列的掩码图像，调色板模式
            
        # make One-hot channel is object index
        # oh_masks.shape[num_frames, H, W, num_objects]
        oh_masks = np.zeros((self.num_frames[video],)+self.shape[video]+(num_objects,), dtype=np.uint8)
        for o in range(num_objects):
            oh_masks[:,:,:,o] = (raw_masks == (o+1)).astype(np.uint8)


        # padding size to be divide by 32
        nf, h, w, _ = oh_masks.shape
        new_h = h + 32 - h % 32
        new_w = w + 32 - w % 32
        # print(new_h, new_w)
        # 对oh_masks和raw_frames进行填充
        lh, uh = (new_h-h) / 2, (new_h-h) / 2 + (new_h-h) % 2
        lw, uw = (new_w-w) / 2, (new_w-w) / 2 + (new_w-w) % 2
        lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)
        pad_masks = np.pad(oh_masks, ((0,0),(lh,uh),(lw,uw),(0,0)), mode='constant')
        pad_frames = np.pad(raw_frames, ((0,0),(lh,uh),(lw,uw),(0,0)), mode='constant')
        info['pad'] = ((lh,uh),(lw,uw))

        th_frames = torch.unsqueeze(torch.from_numpy(np.transpose(pad_frames, (3, 0, 1, 2)).copy()).float(), 0)
        th_masks = torch.unsqueeze(torch.from_numpy(np.transpose(pad_masks, (3, 0, 1, 2)).copy()).long(), 0)
        # 输出的th_frames.shape[1, 3, num_frames, H, W],归一化的数组，每一个值在0-1之间，表示对应像素值大小
        # th_masks.shape[1, num_objects, num_framnes, H, W],one-hot编码后的数组，每一个值为0或1,表示对应的类别，应该从num_objects维度去看。
        return th_frames, th_masks, info