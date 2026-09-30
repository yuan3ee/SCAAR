import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision.transforms import functional as F

###############################################
# Dataset setting
###############################################

DATASET_NAME = 'JL1'

if DATASET_NAME == 'JL1':

    num_classes = 6

    ST_COLORMAP = np.array([
        [255,255,255],
        [0,128,0],
        [0,0,128],
        [128,0,0],
        [0,128,128],
        [128,128,0]
    ])

    ST_CLASSES = [
        'unchanged',
        'farm',
        'road',
        'tree',
        'building',
        'other'
    ]


###############################################
# normalization (和训练一致)
###############################################

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A  = np.array([48.30, 46.27, 48.14])

MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B  = np.array([49.41, 47.01, 47.94])


def normalize_image(im, time='A'):

    if time == 'A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B

    return im


###############################################
# Color convert
###############################################

def Index2Color(label):

    h, w = label.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(len(ST_COLORMAP)):
        color[label == i] = ST_COLORMAP[i]

    return color


###############################################
# Dataset
###############################################

class Data(Dataset):

    def __init__(self, mode):

        root = f'/mnt/sda/su8/SCD/Dataset/JL1_second/{mode}'

        self.im1_dir = os.path.join(root, 'im1')
        self.im2_dir = os.path.join(root, 'im2')
        self.label1_dir = os.path.join(root, 'label1_gray')
        self.label2_dir = os.path.join(root, 'label2_gray')

        self.names = sorted(os.listdir(self.im1_dir))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        name = self.names[idx]

        ###############################################
        # read image
        ###############################################

        img1 = io.imread(os.path.join(self.im1_dir, name))
        img2 = io.imread(os.path.join(self.im2_dir, name))

        ###############################################
        # normalize (和训练完全一致)
        ###############################################

        img1 = normalize_image(img1, 'A')
        img2 = normalize_image(img2, 'B')

        ###############################################
        # to tensor
        ###############################################

        img1 = F.to_tensor(img1)
        img2 = F.to_tensor(img2)

        ###############################################
        # label
        ###############################################

        label_name = name.replace('.tif', '.png')

        label1 = io.imread(os.path.join(self.label1_dir, label_name))
        label2 = io.imread(os.path.join(self.label2_dir, label_name))

        label1 = torch.from_numpy(label1).long()
        label2 = torch.from_numpy(label2).long()

        return img1, img2, label1, label2