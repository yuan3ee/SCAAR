import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from skimage import io


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

root = '/mnt/sda/su8/SCD/Dataset/Landsat-SCD_dataset/test_CR'


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


def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def upsample_to_target(img, target_hw):
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def parse_subset_info(subset_name):
    if subset_name == 'same':
        return 'same', 1.0, 1.0

    if subset_name.startswith('A_low_x'):
        s = float(subset_name.replace('A_low_x', ''))
        return 'A_low', s, 1.0

    if subset_name.startswith('B_low_x'):
        s = float(subset_name.replace('B_low_x', ''))
        return 'B_low', 1.0, s

    raise ValueError(f'Unknown subset name: {subset_name}')


def find_file_by_stem(folder, stem):
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    return None


# =========================
# 离线测试集读取
# =========================
class LandSetAROfflineTestDataset(data.Dataset):
    """
    LandSet 离线测试集

    root:
        /mnt/sda/su8/SCD/Dataset/Landsat-SCD_dataset/test_CR

    subset:
        same
        A_low_x1.5 / A_low_x2.0 / A_low_x3.0 / A_low_x4.0
        B_low_x1.5 / B_low_x2.0 / B_low_x3.0 / B_low_x4.0

    model_type:
        'ours'     -> 直接读取真实异尺寸输入
        'baseline' -> 将低分辨率一侧上采样到标签大小后返回
    """
    def __init__(
        self,
        root='/mnt/sda/su8/SCD/Dataset/Landsat-SCD_dataset/test_CR',
        subset='same',
        model_type='ours',
        return_change=True,
        return_meta=True,
        visual=False,
    ):
        assert model_type in ['ours', 'baseline']

        self.root = root
        self.subset = subset
        self.model_type = model_type
        self.return_change = return_change
        self.return_meta = return_meta
        self.visual = visual

        self.subset_dir = os.path.join(root, subset)
        self.img_A_dir = os.path.join(self.subset_dir, 'A')
        self.img_B_dir = os.path.join(self.subset_dir, 'B')
        self.label_A_dir = os.path.join(self.subset_dir, 'label1')
        self.label_B_dir = os.path.join(self.subset_dir, 'label2')
        self.meta_dir = os.path.join(self.subset_dir, 'meta')

        if not os.path.isdir(self.subset_dir):
            raise FileNotFoundError(f'Not found subset dir: {self.subset_dir}')
        if not os.path.isdir(self.img_A_dir):
            raise FileNotFoundError(f'Not found A dir: {self.img_A_dir}')
        if not os.path.isdir(self.img_B_dir):
            raise FileNotFoundError(f'Not found B dir: {self.img_B_dir}')
        if not os.path.isdir(self.label_A_dir):
            raise FileNotFoundError(f'Not found label1 dir: {self.label_A_dir}')
        if not os.path.isdir(self.label_B_dir):
            raise FileNotFoundError(f'Not found label2 dir: {self.label_B_dir}')

        self.file_list = sorted([
            x for x in os.listdir(self.img_A_dir)
            if x.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])

        if len(self.file_list) == 0:
            raise RuntimeError(f'No image files found in {self.img_A_dir}')

        print(f'Loaded subset [{subset}] with {len(self.file_list)} samples.')

    def __len__(self):
        return len(self.file_list)

    def get_mask_name(self, idx):
        return self.file_list[idx]

    def _load_meta(self, name, img_A_shape=None, img_B_shape=None, label_shape=None):
        meta_path = os.path.join(self.meta_dir, f'{Path(name).stem}.json')
        if os.path.isdir(self.meta_dir) and os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            return meta

        pair_type, scale_A, scale_B = parse_subset_info(self.subset)
        meta = {
            'name': name,
            'pair_type': pair_type,
            'scale_A': float(scale_A),
            'scale_B': float(scale_B),
        }
        if img_A_shape is not None:
            meta['input_size_A'] = list(img_A_shape[:2])
        if img_B_shape is not None:
            meta['input_size_B'] = list(img_B_shape[:2])
        if label_shape is not None:
            meta['target_size'] = list(label_shape[:2])
        return meta

    def __getitem__(self, idx):
        name = self.file_list[idx]
        stem = Path(name).stem

        img_A_path = os.path.join(self.img_A_dir, name)
        img_B_path = find_file_by_stem(self.img_B_dir, stem)
        label_A_path = find_file_by_stem(self.label_A_dir, stem)
        label_B_path = find_file_by_stem(self.label_B_dir, stem)

        if img_B_path is None or label_A_path is None or label_B_path is None:
            raise FileNotFoundError(f'Cannot find matched files for {name}')

        img_A = io.imread(img_A_path)
        img_B = io.imread(img_B_path)
        label_A = io.imread(label_A_path).astype(np.int64)
        label_B = io.imread(label_B_path).astype(np.int64)

        meta = self._load_meta(
            name=name,
            img_A_shape=img_A.shape,
            img_B_shape=img_B.shape,
            label_shape=label_A.shape
        )

        if img_A.ndim == 2:
            img_A = np.stack([img_A] * 3, axis=-1)
        if img_B.ndim == 2:
            img_B = np.stack([img_B] * 3, axis=-1)
        if img_A.shape[-1] > 3:
            img_A = img_A[..., :3]
        if img_B.shape[-1] > 3:
            img_B = img_B[..., :3]

        if self.model_type == 'baseline':
            target_hw = label_A.shape[:2]
            img_A = upsample_to_target(img_A, target_hw)
            img_B = upsample_to_target(img_B, target_hw)

        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')

        change = (label_A != label_B).astype(np.int64)

        img_A = F.to_tensor(img_A)
        img_B = F.to_tensor(img_B)
        label_A = torch.from_numpy(label_A).long()
        label_B = torch.from_numpy(label_B).long()
        change = torch.from_numpy(change).long()

        meta_out = {
            'name': meta.get('name', name),
            'pair_type': meta.get('pair_type', 'unknown'),
            'scale_A': torch.tensor(float(meta.get('scale_A', 1.0)), dtype=torch.float32),
            'scale_B': torch.tensor(float(meta.get('scale_B', 1.0)), dtype=torch.float32),
            'input_size_A': torch.tensor(list(img_A.shape[1:]), dtype=torch.long),
            'input_size_B': torch.tensor(list(img_B.shape[1:]), dtype=torch.long),
            'target_size': torch.tensor(list(label_A.shape), dtype=torch.long),
            'subset': self.subset,
            'model_type': self.model_type,
        }

        out = [img_A, img_B, label_A, label_B]

        if self.return_change:
            out.append(change)

        if self.return_meta:
            out.append(meta_out)

        if self.visual:
            out.append(name)

        return tuple(out)


# =========================
# collate_fn
# =========================
def collate_fn_ours(batch):
    sample_len = len(batch[0])

    has_change = sample_len >= 5
    has_meta = sample_len >= 6
    has_name = sample_len >= 7

    img_As = [b[0] for b in batch]
    img_Bs = [b[1] for b in batch]
    label_As = torch.stack([b[2] for b in batch], dim=0)
    label_Bs = torch.stack([b[3] for b in batch], dim=0)

    out = [img_As, img_Bs, label_As, label_Bs]

    cursor = 4
    if has_change:
        changes = torch.stack([b[cursor] for b in batch], dim=0)
        out.append(changes)
        cursor += 1

    if has_meta:
        metas = [b[cursor] for b in batch]
        out.append(metas)
        cursor += 1

    if has_name:
        names = [b[cursor] for b in batch]
        out.append(names)

    return tuple(out)


def collate_fn_baseline(batch):
    sample_len = len(batch[0])

    has_change = sample_len >= 5
    has_meta = sample_len >= 6
    has_name = sample_len >= 7

    img_As = torch.stack([b[0] for b in batch], dim=0)
    img_Bs = torch.stack([b[1] for b in batch], dim=0)
    label_As = torch.stack([b[2] for b in batch], dim=0)
    label_Bs = torch.stack([b[3] for b in batch], dim=0)

    out = [img_As, img_Bs, label_As, label_Bs]

    cursor = 4
    if has_change:
        changes = torch.stack([b[cursor] for b in batch], dim=0)
        out.append(changes)
        cursor += 1

    if has_meta:
        metas = [b[cursor] for b in batch]
        out.append(metas)
        cursor += 1

    if has_name:
        names = [b[cursor] for b in batch]
        out.append(names)

    return tuple(out)


# =========================
# 单个测试子集 loader
# =========================
def build_single_test_loader(
    subset='same',
    model_type='ours',
    batch_size=4,
    num_workers=4,
    return_change=True,
    return_meta=True,
    visual=False,
):
    dataset = LandSetAROfflineTestDataset(
        root='/mnt/sda/su8/SCD/Dataset/Landsat-SCD_dataset/test_CR',
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
        collate_fn=collate_fn,
    )
    return loader


# =========================
# 直接给两个常用 loader 写法
# =========================
test_loader_ours = build_single_test_loader(
    subset='same',
    model_type='ours',
    batch_size=4,
    num_workers=4,
    return_change=False,
    return_meta=False,
    visual=False,
)

test_loader_baseline = build_single_test_loader(
    subset='same',
    model_type='baseline',
    batch_size=4,
    num_workers=4,
    return_change=False,
    return_meta=False,
    visual=False,
)