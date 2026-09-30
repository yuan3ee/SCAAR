import os
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from skimage import io
import cv2


# =========================
# LandSet-SCD
# =========================
num_classes = 5
ST_COLORMAP = [
    [255, 255, 255],   # unchanged
    [0, 155, 0],       # farmland
    [255, 165, 0],     # desert
    [230, 30, 100],    # building
    [0, 170, 240],     # water
]
ST_CLASSES = ['unchanged', 'farmland', 'desert', 'building', 'water']

# MEAN_A = np.array([113.40, 114.08, 116.45], dtype=np.float32)
# STD_A  = np.array([48.30,  46.27,  48.14], dtype=np.float32)
# MEAN_B = np.array([111.07, 114.04, 118.18], dtype=np.float32)
# STD_B  = np.array([49.41,  47.01,  47.94], dtype=np.float32)

MEAN_A = np.array([141.53, 139.20, 137.73], dtype=np.float32)
STD_A  = np.array([81.99, 83.31, 83.89], dtype=np.float32)
MEAN_B = np.array([137.36, 136.50, 135.144], dtype=np.float32)
STD_B  = np.array([85.97, 86.01, 86.81], dtype=np.float32)

def normalize_image(img, time='A'):
    img = img.astype(np.float32)
    if time == 'A':
        img = (img - MEAN_A) / STD_A
    else:
        img = (img - MEAN_B) / STD_B
    return img


def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def read_val_paths(root):
    """
    固定混合验证集结构:
        val_CR/
            A/
            B/
            label1/
            label2/
            meta/
            split_info.json
    """
    A_dir = os.path.join(root, 'A')
    B_dir = os.path.join(root, 'B')
    label1_dir = os.path.join(root, 'label1')
    label2_dir = os.path.join(root, 'label2')
    meta_dir = os.path.join(root, 'meta')

    if not os.path.isdir(A_dir):
        raise FileNotFoundError(f'Cannot find {A_dir}')
    if not os.path.isdir(B_dir):
        raise FileNotFoundError(f'Cannot find {B_dir}')
    if not os.path.isdir(label1_dir):
        raise FileNotFoundError(f'Cannot find {label1_dir}')
    if not os.path.isdir(label2_dir):
        raise FileNotFoundError(f'Cannot find {label2_dir}')
    if not os.path.isdir(meta_dir):
        raise FileNotFoundError(f'Cannot find {meta_dir}')

    file_list = sorted([
        x for x in os.listdir(A_dir)
        if x.lower().endswith('.png')
    ])

    imgs_list_A, imgs_list_B = [], []
    labels_A, labels_B, metas = [], [], []

    for name in file_list:
        img_A_path = os.path.join(A_dir, name)
        img_B_path = os.path.join(B_dir, name)
        label_A_path = os.path.join(label1_dir, name)
        label_B_path = os.path.join(label2_dir, name)
        meta_path = os.path.join(meta_dir, f'{Path(name).stem}.json')

        if not os.path.exists(img_B_path):
            raise FileNotFoundError(f'Cannot find matched B image for {name}')
        if not os.path.exists(label_A_path):
            raise FileNotFoundError(f'Cannot find matched label1 for {name}')
        if not os.path.exists(label_B_path):
            raise FileNotFoundError(f'Cannot find matched label2 for {name}')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f'Cannot find meta for {name}')

        imgs_list_A.append(img_A_path)
        imgs_list_B.append(img_B_path)
        labels_A.append(label_A_path)
        labels_B.append(label_B_path)
        metas.append(meta_path)

    print(f'{len(file_list)} val images loaded.')
    return imgs_list_A, imgs_list_B, labels_A, labels_B, metas


class LandSetARValDataset(Dataset):
    """
    固定混合验证集

    model_type:
        - 'ours'     : 保持真实异尺寸输入
        - 'baseline' : 将低分辨率一侧上采样到标签对应的大尺寸网格
    """
    def __init__(
        self,
        root='/mnt/sda/su8/SCD/Dataset/Landsat-SCD_dataset/val_CR',
        model_type='ours',
        return_change=True,
        return_meta=True,
        visual=False,
    ):
        assert model_type in ['ours', 'baseline']

        self.root = root
        self.model_type = model_type
        self.return_change = return_change
        self.return_meta = return_meta
        self.visual = visual

        self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B, self.meta_list = \
            read_val_paths(root)

    def __len__(self):
        return len(self.imgs_list_A)

    def get_mask_name(self, idx):
        return os.path.split(self.imgs_list_A[idx])[-1]

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx])
        img_B = io.imread(self.imgs_list_B[idx])
        label_A = io.imread(self.labels_A[idx])
        label_B = io.imread(self.labels_B[idx])

        with open(self.meta_list[idx], 'r', encoding='utf-8') as f:
            meta = json.load(f)

        name = self.get_mask_name(idx)

        # 保证是 RGB
        if img_A.ndim == 2:
            img_A = np.stack([img_A] * 3, axis=-1)
        if img_B.ndim == 2:
            img_B = np.stack([img_B] * 3, axis=-1)
        if img_A.shape[-1] > 3:
            img_A = img_A[..., :3]
        if img_B.shape[-1] > 3:
            img_B = img_B[..., :3]

        # baseline: 将低分辨率一侧上采样回标签对应的大尺寸网格
        if self.model_type == 'baseline':
            target_h, target_w = label_A.shape[:2]

            if img_A.shape[:2] != (target_h, target_w):
                img_A = cv2.resize(img_A, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

            if img_B.shape[:2] != (target_h, target_w):
                img_B = cv2.resize(img_B, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # 归一化
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')

        # 标签与变化图
        label_A = label_A.astype(np.int64)
        label_B = label_B.astype(np.int64)
        change = (label_A != label_B).astype(np.int64)

        # tensor
        img_A_tensor = F.to_tensor(img_A)
        img_B_tensor = F.to_tensor(img_B)
        label_A_tensor = torch.from_numpy(label_A).long()
        label_B_tensor = torch.from_numpy(label_B).long()
        change_tensor = torch.from_numpy(change).long()

        meta_out = {
            'name': name,
            'pair_type': meta.get('pair_type', 'unknown'),
            'scale_A': torch.tensor(float(meta.get('scale_A', 1.0)), dtype=torch.float32),
            'scale_B': torch.tensor(float(meta.get('scale_B', 1.0)), dtype=torch.float32),
            'input_size_A': torch.tensor(img_A_tensor.shape[-2:], dtype=torch.long),
            'input_size_B': torch.tensor(img_B_tensor.shape[-2:], dtype=torch.long),
            'target_size': torch.tensor(label_A.shape, dtype=torch.long),
        }

        out = [img_A_tensor, img_B_tensor, label_A_tensor, label_B_tensor]

        if self.return_change:
            out.append(change_tensor)
        if self.return_meta:
            out.append(meta_out)
        if self.visual:
            out.append(name)

        return tuple(out)


def collate_fn_ours(batch):
    img_As = [b[0] for b in batch]
    img_Bs = [b[1] for b in batch]
    label_As = torch.stack([b[2] for b in batch], dim=0)
    label_Bs = torch.stack([b[3] for b in batch], dim=0)

    out = [img_As, img_Bs, label_As, label_Bs]

    if len(batch[0]) >= 5:
        changes = torch.stack([b[4] for b in batch], dim=0)
        out.append(changes)
    if len(batch[0]) >= 6:
        metas = [b[5] for b in batch]
        out.append(metas)
    if len(batch[0]) >= 7:
        names = [b[6] for b in batch]
        out.append(names)

    return tuple(out)


def collate_fn_baseline(batch):
    img_As = torch.stack([b[0] for b in batch], dim=0)
    img_Bs = torch.stack([b[1] for b in batch], dim=0)
    label_As = torch.stack([b[2] for b in batch], dim=0)
    label_Bs = torch.stack([b[3] for b in batch], dim=0)

    out = [img_As, img_Bs, label_As, label_Bs]

    if len(batch[0]) >= 5:
        changes = torch.stack([b[4] for b in batch], dim=0)
        out.append(changes)
    if len(batch[0]) >= 6:
        metas = [b[5] for b in batch]
        out.append(metas)
    if len(batch[0]) >= 7:
        names = [b[6] for b in batch]
        out.append(names)

    return tuple(out)


def build_val_loader(
    root='/mnt/sda/su8/SCD/Dataset/Landsat-SCD_dataset/val_CR',
    model_type='ours',
    batch_size=4,
    num_workers=4,
    return_change=True,
    return_meta=True,
    visual=False,
):
    dataset = LandSetARValDataset(
        root=root,
        model_type=model_type,
        return_change=return_change,
        return_meta=return_meta,
        visual=visual,
    )

    collate_fn = collate_fn_ours if model_type == 'ours' else collate_fn_baseline

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return loader

val_loader_ours = build_val_loader(
    root='/mnt/sda/su8/SCD/Dataset/Landsat-SCD_dataset/val_CR',
    model_type='ours',
    batch_size=4,
    num_workers=4,
    return_change=False,
    return_meta=False,
    visual=False,
)

val_loader_baseline = build_val_loader(
    root='/mnt/sda/su8/SCD/Dataset/Landsat-SCD_dataset/val_CR',
    model_type='baseline',
    batch_size=4,
    num_workers=4,
    return_change=False,
    return_meta=False,
    visual=False,
)