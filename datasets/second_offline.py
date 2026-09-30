import os
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from skimage import io
import cv2


# =========================
# SECOND
# =========================
num_classes = 7
ST_COLORMAP = [
    [255, 255, 255],   # unchanged
    [0, 0, 255],       # water
    [128, 128, 128],   # ground
    [0, 128, 0],       # low vegetation
    [0, 255, 0],       # tree
    [128, 0, 0],       # building
    [255, 0, 0],       # sports field
]
ST_CLASSES = [
    'unchanged', 'water', 'ground', 'low vegetation',
    'tree', 'building', 'sports field'
]

MEAN_A = np.array([113.40, 114.08, 116.45], dtype=np.float32)
STD_A  = np.array([48.30,  46.27,  48.14], dtype=np.float32)
MEAN_B = np.array([111.07, 114.04, 118.18], dtype=np.float32)
STD_B  = np.array([49.41,  47.01,  47.94], dtype=np.float32)

# 离线测试集根目录
root = '/mnt/sda/su8/SCD/Dataset/second/test_CR'


# =========================
# 工具函数
# =========================
def normalize_image(img, time='A'):
    img = img.astype(np.float32)
    if time == 'A':
        img = (img - MEAN_A) / STD_A
    else:
        img = (img - MEAN_B) / STD_B
    return img


def read_offline_subset_paths(subset):
    """
    离线测试集结构:
        root/
            same/
                im1, im2, label1, label2, meta
            A_low_x2.0/
                im1, im2, label1, label2, meta
            B_low_x4.0/
                ...
    """
    subset_dir = os.path.join(root, subset)
    img_A_dir = os.path.join(subset_dir, 'im1')
    img_B_dir = os.path.join(subset_dir, 'im2')
    label_A_dir = os.path.join(subset_dir, 'label1')
    label_B_dir = os.path.join(subset_dir, 'label2')
    meta_dir = os.path.join(subset_dir, 'meta')

    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(f'Cannot find subset dir: {subset_dir}')
    if not os.path.isdir(img_A_dir):
        raise FileNotFoundError(f'Cannot find im1 dir: {img_A_dir}')
    if not os.path.isdir(img_B_dir):
        raise FileNotFoundError(f'Cannot find im2 dir: {img_B_dir}')
    if not os.path.isdir(label_A_dir):
        raise FileNotFoundError(f'Cannot find label1 dir: {label_A_dir}')
    if not os.path.isdir(label_B_dir):
        raise FileNotFoundError(f'Cannot find label2 dir: {label_B_dir}')

    data_list = sorted(os.listdir(img_A_dir))

    imgs_list_A, imgs_list_B = [], []
    labels_A, labels_B = [], []
    metas = []

    count = 0
    for it in data_list:
        if it.endswith('.png') or it.endswith('.tif') or it.endswith('.tiff') or it.endswith('.jpg') or it.endswith('.jpeg'):
            stem = Path(it).stem

            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)

            label_A_path = None
            label_B_path = None
            for ext in ['.png', '.tif', '.tiff']:
                pa = os.path.join(label_A_dir, stem + ext)
                pb = os.path.join(label_B_dir, stem + ext)
                if os.path.exists(pa):
                    label_A_path = pa
                if os.path.exists(pb):
                    label_B_path = pb

            if label_A_path is None or label_B_path is None:
                raise FileNotFoundError(f'Cannot find label for {it} in subset {subset}')

            meta_path = os.path.join(meta_dir, f'{stem}.json') if os.path.isdir(meta_dir) else None

            imgs_list_A.append(img_A_path)
            imgs_list_B.append(img_B_path)
            labels_A.append(label_A_path)
            labels_B.append(label_B_path)
            metas.append(meta_path)

        count += 1
        if not count % 500:
            print('%d/%d images scanned.' % (count, len(data_list)))

    print(f'{len(imgs_list_A)} offline test images loaded from subset [{subset}].')
    return imgs_list_A, imgs_list_B, labels_A, labels_B, metas


def infer_meta_from_subset(subset, name):
    """
    如果 meta 文件不存在，则从 subset 名称推断
    """
    meta = {'name': name}

    if subset == 'same':
        meta['pair_type'] = 'same'
        meta['scale_A'] = 1.0
        meta['scale_B'] = 1.0
    elif subset.startswith('A_low_x'):
        s = float(subset.replace('A_low_x', ''))
        meta['pair_type'] = 'A_low'
        meta['scale_A'] = s
        meta['scale_B'] = 1.0
    elif subset.startswith('B_low_x'):
        s = float(subset.replace('B_low_x', ''))
        meta['pair_type'] = 'B_low'
        meta['scale_A'] = 1.0
        meta['scale_B'] = s
    else:
        raise ValueError(f'Unknown subset name: {subset}')

    return meta


# =========================
# 数据集
# =========================
class SecondAROfflineTestDataset(Dataset):
    """
    离线测试集

    model_type:
        - 'ours'     : 输入保持真实异尺寸
        - 'baseline' : 将低分辨率一侧上采样到标签对应的大尺寸网格

    subset:
        - same
        - A_low_x1.5 / A_low_x2.0 / A_low_x3.0 / A_low_x4.0
        - B_low_x1.5 / B_low_x2.0 / B_low_x3.0 / B_low_x4.0
    """
    def __init__(
        self,
        subset='same',
        model_type='ours',
        return_change=True,
        return_meta=True,
        visual=False,
    ):
        assert model_type in ['ours', 'baseline']

        self.subset = subset
        self.model_type = model_type
        self.return_change = return_change
        self.return_meta = return_meta
        self.visual = visual

        self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B, self.meta_list = \
            read_offline_subset_paths(subset)

    def __len__(self):
        return len(self.imgs_list_A)

    def get_mask_name(self, idx):
        return os.path.split(self.imgs_list_A[idx])[-1]

    def __getitem__(self, idx):
        # 1. 读取图像和标签
        img_A = io.imread(self.imgs_list_A[idx])
        img_B = io.imread(self.imgs_list_B[idx])
        label_A = io.imread(self.labels_A[idx])
        label_B = io.imread(self.labels_B[idx])

        name = self.get_mask_name(idx)

        # 2. 读取 meta
        meta_path = self.meta_list[idx]
        if meta_path is not None and os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        else:
            meta = infer_meta_from_subset(self.subset, name)

        # 3. 保证图像是 RGB
        if img_A.ndim == 2:
            img_A = np.stack([img_A] * 3, axis=-1)
        if img_B.ndim == 2:
            img_B = np.stack([img_B] * 3, axis=-1)
        if img_A.shape[-1] > 3:
            img_A = img_A[..., :3]
        if img_B.shape[-1] > 3:
            img_B = img_B[..., :3]

        # 4. baseline: 将低分辨率侧上采样到标签对应的大尺寸网格
        if self.model_type == 'baseline':
            target_h, target_w = label_A.shape[:2]

            if img_A.shape[:2] != (target_h, target_w):
                img_A = cv2.resize(img_A, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

            if img_B.shape[:2] != (target_h, target_w):
                img_B = cv2.resize(img_B, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # 5. 归一化
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')

        # 6. 标签和 change
        label_A = label_A.astype(np.int64)
        label_B = label_B.astype(np.int64)
        change = (label_A != label_B).astype(np.int64)

        # 7. tensor
        img_A_tensor = F.to_tensor(img_A)
        img_B_tensor = F.to_tensor(img_B)
        label_A_tensor = torch.from_numpy(label_A).long()
        label_B_tensor = torch.from_numpy(label_B).long()
        change_tensor = torch.from_numpy(change).long()

        meta_out = {
            'scale_A': torch.tensor(float(meta.get('scale_A', 1.0)), dtype=torch.float32),
            'scale_B': torch.tensor(float(meta.get('scale_B', 1.0)), dtype=torch.float32),
            'pair_type': meta.get('pair_type', 'unknown'),
            'input_size_A': torch.tensor(img_A_tensor.shape[-2:], dtype=torch.long),
            'input_size_B': torch.tensor(img_B_tensor.shape[-2:], dtype=torch.long),
            'target_size': torch.tensor(label_A.shape, dtype=torch.long),
            'name': name,
            'subset': self.subset,
        }

        out = [img_A_tensor, img_B_tensor, label_A_tensor, label_B_tensor]

        if self.return_change:
            out.append(change_tensor)
        if self.return_meta:
            out.append(meta_out)
        if self.visual:
            out.append(name)

        return tuple(out)


# =========================
# collate_fn
# =========================
def collate_fn_ours(batch):
    """
    用于你的方法：
    A/B 可能不同尺寸，不能直接 stack
    返回 list(img_A), list(img_B), 以及 stack 后的标签
    """
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
    """
    用于对比方法：
    A/B 已经被统一到同一尺寸，可以直接 stack
    """
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

def build_single_test_loader(
    subset='same',
    model_type='ours',
    batch_size=4,
    num_workers=4,
    return_change=True,
    return_meta=True,
    visual=False,
):
    """
    构建单个测试子集的 DataLoader

    subset:
        same
        A_low_x1.5 / A_low_x2.0 / A_low_x3.0 / A_low_x4.0
        B_low_x1.5 / B_low_x2.0 / B_low_x3.0 / B_low_x4.0

    model_type:
        'ours' or 'baseline'
    """
    dataset = SecondAROfflineTestDataset(
        subset=subset,
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

test_loader_ours = build_single_test_loader(
    subset='B_low_x2.0',
    model_type='ours',
    batch_size=4,
    num_workers=4,
    return_change=True,
    return_meta=True,
    visual=False,
)

test_loader_baseline = build_single_test_loader(
    subset='B_low_x2.0',
    model_type='baseline',
    batch_size=4,
    num_workers=4,
    return_change=True,
    return_meta=True,
    visual=False,
)


