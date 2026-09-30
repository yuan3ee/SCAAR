import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
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

ST_CLASSES = [
    'unchanged',
    'farmland',
    'desert',
    'building',
    'water'
]

MEAN_A = np.array([141.53, 139.20, 137.73], dtype=np.float32)
STD_A  = np.array([81.99, 83.31, 83.89], dtype=np.float32)

MEAN_B = np.array([137.36, 136.50, 135.144], dtype=np.float32)
STD_B  = np.array([85.97, 86.01, 86.81], dtype=np.float32)

root = '/mnt/sda/su8/SCD/Dataset/Landsat-SCD_dataset'


# =========================
# 工具函数
# =========================
colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap


def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def normalize_image(img, time='A'):
    img = img.astype(np.float32)

    if time == 'A':
        img = (img - MEAN_A) / STD_A
    else:
        img = (img - MEAN_B) / STD_B

    return img


def rand_rot90_flip_MCD(img_A, img_B, label_A, label_B):
    """
    图像与标签同步随机旋转翻转。
    """
    k = np.random.randint(0, 4)

    img_A = np.rot90(img_A, k).copy()
    img_B = np.rot90(img_B, k).copy()

    label_A = np.rot90(label_A, k).copy()
    label_B = np.rot90(label_B, k).copy()

    if np.random.rand() < 0.5:
        img_A = np.fliplr(img_A).copy()
        img_B = np.fliplr(img_B).copy()

        label_A = np.fliplr(label_A).copy()
        label_B = np.fliplr(label_B).copy()

    if np.random.rand() < 0.5:
        img_A = np.flipud(img_A).copy()
        img_B = np.flipud(img_B).copy()

        label_A = np.flipud(label_A).copy()
        label_B = np.flipud(label_B).copy()

    return img_A, img_B, label_A, label_B


def read_RSimages_paths(mode):
    img_A_dir = os.path.join(root, mode, 'A')
    img_B_dir = os.path.join(root, mode, 'B')
    label_A_dir = os.path.join(root, mode, 'label1')
    label_B_dir = os.path.join(root, mode, 'label2')

    data_list = sorted(os.listdir(img_A_dir))

    imgs_list_A = []
    imgs_list_B = []
    labels_A = []
    labels_B = []

    count = 0

    for it in data_list:
        if it.endswith('.png'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)

            label_A_path = os.path.join(label_A_dir, it)
            label_B_path = os.path.join(label_B_dir, it)

            if not os.path.exists(img_B_path):
                raise FileNotFoundError(f'Cannot find matched B image for {it}')

            if not os.path.exists(label_A_path):
                raise FileNotFoundError(f'Cannot find label1 for {it}')

            if not os.path.exists(label_B_path):
                raise FileNotFoundError(f'Cannot find label2 for {it}')

            imgs_list_A.append(img_A_path)
            imgs_list_B.append(img_B_path)

            labels_A.append(label_A_path)
            labels_B.append(label_B_path)

        count += 1

        if not count % 500:
            print('%d/%d images scanned.' % (count, len(data_list)))

    print(f'{len(imgs_list_A)} {mode} images loaded.')
    return imgs_list_A, imgs_list_B, labels_A, labels_B


def sample_scale_log_uniform(scale_min=1.0, scale_max=4.0):
    """
    从 [scale_min, scale_max] 对数均匀采样倍率。
    """
    if scale_min == scale_max:
        return float(scale_min)

    log_s = np.random.uniform(np.log(scale_min), np.log(scale_max))
    return float(np.exp(log_s))


def gaussian_blur(img, sigma):
    if sigma <= 1e-6:
        return img

    ksize = max(3, int(2 * round(3 * sigma) + 1))

    if ksize % 2 == 0:
        ksize += 1

    return cv2.GaussianBlur(
        img,
        (ksize, ksize),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101
    )


def degrade_resolution(img, scale, keep_input_size=False, use_blur=True):
    """
    分辨率退化：
        1. 先模糊；
        2. 再下采样；
        3. 若 keep_input_size=True，则再插值回原始尺寸。

    model_type='ours':
        keep_input_size=False，返回原生低分辨率图像。

    model_type='baseline':
        keep_input_size=True，返回上采样回原始尺寸的图像。
    """
    img = img.astype(np.uint8)

    if scale <= 1.0 + 1e-6:
        return img.copy()

    h, w = img.shape[:2]

    if use_blur:
        sigma = 0.35 + 0.30 * (scale - 1.0)
        img = gaussian_blur(img, sigma=sigma)

    low_h = max(1, int(round(h / scale)))
    low_w = max(1, int(round(w / scale)))

    img_low = cv2.resize(
        img,
        (low_w, low_h),
        interpolation=cv2.INTER_AREA
    )

    if keep_input_size:
        img_out = cv2.resize(
            img_low,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )
        return img_out

    return img_low


def sample_pair_by_prob(
    p_same=0.30,
    p_A_low=0.35,
    p_B_low=0.35,
    scale_min=1.0,
    scale_max=4.0
):
    """
    按 batch 级别采样 pair_type 和 scale。

    pair_type:
        0: same,  scale_A=1, scale_B=1
        1: A_low, scale_A=s, scale_B=1
        2: B_low, scale_A=1, scale_B=s
    """
    prob_sum = p_same + p_A_low + p_B_low

    if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError(
            f'p_same + p_A_low + p_B_low must be 1.0, got {prob_sum}'
        )

    u = np.random.rand()

    if u < p_same:
        pair_type = 0
        scale_A = 1.0
        scale_B = 1.0

    elif u < p_same + p_A_low:
        pair_type = 1
        scale_A = sample_scale_log_uniform(scale_min, scale_max)
        scale_B = 1.0

    else:
        pair_type = 2
        scale_A = 1.0
        scale_B = sample_scale_log_uniform(scale_min, scale_max)

    return pair_type, scale_A, scale_B


def maybe_color_to_index(label):
    """
    兼容两种标签：
        1. H x W 的 index label；
        2. H x W x 3 的 color label。

    原 Landset_online.py 里定义了 Color2Index，但 __getitem__ 未调用。
    这里做兼容处理，避免彩色标签时训练异常。
    """
    if label.ndim == 3:
        label = Color2Index(label).astype(np.int64)
    else:
        label = label.astype(np.int64)

    return label


# =========================
# Dataset
# =========================
class LandSetARDataset(Dataset):
    """
    Resolution-arbitrary LandSet-SCD dataset.

    model_type:
        - 'ours':
            输入可以不同尺寸，不上采样图像；
        - 'baseline':
            退化后再上采样回原始尺寸，用于上采样基线。

    pair_type:
        0: (1, 1)
        1: (s, 1), A degraded
        2: (1, s), B degraded

    关键：
        为支持 batch_size > 1 且同 batch 内倍率一致，
        __getitem__ 支持两种 index：

        1. int:
            常规单样本随机采样，兼容旧代码；

        2. tuple/list:
            (idx, pair_type, scale_A, scale_B)
            由 SameScaleBatchSampler 生成，保证同 batch 内 pair_type 和 scale 一致。
    """
    def __init__(
        self,
        mode='train',
        random_flip=False,
        model_type='ours',
        scale_min=1.0,
        scale_max=4.0,
        use_blur=True,

        p_same=0.30,
        p_A_low=0.35,
        p_B_low=0.35,

        eval_pair_type=0,
        eval_scale_A=1.0,
        eval_scale_B=1.0,

        return_change=True,
        return_meta=True,
        visual=False,
    ):
        assert model_type in ['ours', 'baseline']
        assert mode in ['train', 'val', 'test']

        self.mode = mode
        self.random_flip = random_flip and mode == 'train'
        self.model_type = model_type

        self.scale_min = scale_min
        self.scale_max = scale_max
        self.use_blur = use_blur

        self.p_same = p_same
        self.p_A_low = p_A_low
        self.p_B_low = p_B_low

        prob_sum = p_same + p_A_low + p_B_low

        if abs(prob_sum - 1.0) > 1e-6:
            raise ValueError(
                f'p_same + p_A_low + p_B_low must be 1.0, got {prob_sum}'
            )

        self.eval_pair_type = eval_pair_type
        self.eval_scale_A = eval_scale_A
        self.eval_scale_B = eval_scale_B

        self.return_change = return_change
        self.return_meta = return_meta
        self.visual = visual

        self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B = read_RSimages_paths(mode)

    def __len__(self):
        return len(self.imgs_list_A)

    def get_mask_name(self, idx):
        return os.path.split(self.imgs_list_A[idx])[-1]

    def _sample_train_pair(self):
        return sample_pair_by_prob(
            p_same=self.p_same,
            p_A_low=self.p_A_low,
            p_B_low=self.p_B_low,
            scale_min=self.scale_min,
            scale_max=self.scale_max
        )

    def _get_eval_pair(self):
        return self.eval_pair_type, self.eval_scale_A, self.eval_scale_B

    def _parse_index(self, index):
        """
        支持：
            index = int
            index = (idx, pair_type, scale_A, scale_B)
        """
        if isinstance(index, (tuple, list)):
            if len(index) != 4:
                raise ValueError(
                    'When index is tuple/list, it must be '
                    '(idx, pair_type, scale_A, scale_B).'
                )

            idx = int(index[0])
            pair_type = int(index[1])
            scale_A = float(index[2])
            scale_B = float(index[3])

            return idx, pair_type, scale_A, scale_B

        idx = int(index)

        if self.mode == 'train':
            pair_type, scale_A, scale_B = self._sample_train_pair()
        else:
            pair_type, scale_A, scale_B = self._get_eval_pair()

        return idx, pair_type, scale_A, scale_B

    def __getitem__(self, index):
        idx, pair_type, scale_A, scale_B = self._parse_index(index)

        # 1. 读取原始图像和标签
        img_A = io.imread(self.imgs_list_A[idx])
        img_B = io.imread(self.imgs_list_B[idx])

        label_A = io.imread(self.labels_A[idx])
        label_B = io.imread(self.labels_B[idx])

        if img_A.ndim == 2:
            img_A = np.stack([img_A] * 3, axis=-1)

        if img_B.ndim == 2:
            img_B = np.stack([img_B] * 3, axis=-1)

        if img_A.shape[-1] > 3:
            img_A = img_A[..., :3]

        if img_B.shape[-1] > 3:
            img_B = img_B[..., :3]

        # 标签兼容：彩色标签 -> index label
        label_A = maybe_color_to_index(label_A)
        label_B = maybe_color_to_index(label_B)

        # 2. 几何增强
        if self.random_flip:
            img_A, img_B, label_A, label_B = rand_rot90_flip_MCD(
                img_A,
                img_B,
                label_A,
                label_B
            )

        # 3. 图像退化
        if self.model_type == 'ours':
            # CRCI-Net:
            # 保持真实不同输入尺寸，不在图像域上采样
            img_A = degrade_resolution(
                img_A,
                scale_A,
                keep_input_size=False,
                use_blur=self.use_blur
            )

            img_B = degrade_resolution(
                img_B,
                scale_B,
                keep_input_size=False,
                use_blur=self.use_blur
            )

        else:
            # Baseline:
            # 退化后再插值回原始尺寸
            img_A = degrade_resolution(
                img_A,
                scale_A,
                keep_input_size=True,
                use_blur=self.use_blur
            )

            img_B = degrade_resolution(
                img_B,
                scale_B,
                keep_input_size=True,
                use_blur=self.use_blur
            )

        # 4. 归一化
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')

        # 5. 标签保持原始高分辨率网格
        label_A = label_A.astype(np.int64)
        label_B = label_B.astype(np.int64)

        change = (label_A != label_B).astype(np.int64)

        # 6. tensor
        img_A_tensor = F.to_tensor(img_A)
        img_B_tensor = F.to_tensor(img_B)

        label_A_tensor = torch.from_numpy(label_A).long()
        label_B_tensor = torch.from_numpy(label_B).long()
        change_tensor = torch.from_numpy(change).long()

        meta = {
            'scale_A': torch.tensor(scale_A, dtype=torch.float32),
            'scale_B': torch.tensor(scale_B, dtype=torch.float32),
            'pair_type': torch.tensor(pair_type, dtype=torch.long),
            'input_size_A': torch.tensor(img_A_tensor.shape[-2:], dtype=torch.long),
            'input_size_B': torch.tensor(img_B_tensor.shape[-2:], dtype=torch.long),
            'target_size': torch.tensor(label_A.shape, dtype=torch.long),
            'name': self.get_mask_name(idx),
        }

        out = [
            img_A_tensor,
            img_B_tensor,
            label_A_tensor,
            label_B_tensor
        ]

        if self.return_change:
            out.append(change_tensor)

        if self.return_meta:
            out.append(meta)

        if self.visual:
            out.append(self.get_mask_name(idx))

        return tuple(out)


# =========================
# Same-scale batch sampler
# =========================
class SameScaleBatchSampler(Sampler):
    """
    同一个 batch 内倍率一致的 BatchSampler。

    作用：
        1. 每个 batch 采样一个统一的 pair_type / scale_A / scale_B；
        2. batch 内所有样本使用相同退化方向和倍率；
        3. 因此 img_A / img_B 在 batch 内分别具有一致尺寸，可以默认 stack；
        4. 适用于 model_type='ours' 且 train_batch_size > 1。

    注意：
        使用该 sampler 时，DataLoader 中不要再设置 batch_size / shuffle / drop_last。
    """
    def __init__(
        self,
        dataset,
        batch_size=4,
        drop_last=True,
        shuffle=True,

        scale_min=1.0,
        scale_max=4.0,
        p_same=0.30,
        p_A_low=0.35,
        p_B_low=0.35,
    ):
        if not isinstance(dataset, LandSetARDataset):
            raise TypeError('dataset must be an instance of LandSetARDataset.')

        if batch_size < 1:
            raise ValueError('batch_size must be >= 1.')

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.scale_min = scale_min
        self.scale_max = scale_max

        self.p_same = p_same
        self.p_A_low = p_A_low
        self.p_B_low = p_B_low

        prob_sum = p_same + p_A_low + p_B_low

        if abs(prob_sum - 1.0) > 1e-6:
            raise ValueError(
                f'p_same + p_A_low + p_B_low must be 1.0, got {prob_sum}'
            )

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        batch = []

        for idx in indices:
            batch.append(idx)

            if len(batch) == self.batch_size:
                pair_type, scale_A, scale_B = sample_pair_by_prob(
                    p_same=self.p_same,
                    p_A_low=self.p_A_low,
                    p_B_low=self.p_B_low,
                    scale_min=self.scale_min,
                    scale_max=self.scale_max
                )

                yield [
                    (sample_idx, pair_type, scale_A, scale_B)
                    for sample_idx in batch
                ]

                batch = []

        if len(batch) > 0 and not self.drop_last:
            pair_type, scale_A, scale_B = sample_pair_by_prob(
                p_same=self.p_same,
                p_A_low=self.p_A_low,
                p_B_low=self.p_B_low,
                scale_min=self.scale_min,
                scale_max=self.scale_max
            )

            yield [
                (sample_idx, pair_type, scale_A, scale_B)
                for sample_idx in batch
            ]

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size

        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# =========================
# collate_fn
# =========================
def collate_fn_ours_list(batch):
    """
    备用方案：
        如果不使用 SameScaleBatchSampler，batch 内仍可能尺寸不同，
        可以使用该 collate_fn 返回 list。

    当前“同 batch 倍率一致”方案一般不需要这个函数。
    """
    first = batch[0]
    sample_len = len(first)

    img_As = [item[0] for item in batch]
    img_Bs = [item[1] for item in batch]

    label_As = torch.stack([item[2] for item in batch], dim=0)
    label_Bs = torch.stack([item[3] for item in batch], dim=0)

    out = [img_As, img_Bs, label_As, label_Bs]

    cursor = 4

    if sample_len > cursor and torch.is_tensor(first[cursor]):
        changes = torch.stack([item[cursor] for item in batch], dim=0)
        out.append(changes)
        cursor += 1

    if sample_len > cursor and isinstance(first[cursor], dict):
        metas = [item[cursor] for item in batch]
        out.append(metas)
        cursor += 1

    if sample_len > cursor:
        names = [item[cursor] for item in batch]
        out.append(names)

    return tuple(out)


def collate_fn_stack(batch):
    """
    显式 stack 版本。
    同 batch 倍率一致时使用。
    """
    first = batch[0]
    sample_len = len(first)

    img_As = torch.stack([item[0] for item in batch], dim=0)
    img_Bs = torch.stack([item[1] for item in batch], dim=0)

    label_As = torch.stack([item[2] for item in batch], dim=0)
    label_Bs = torch.stack([item[3] for item in batch], dim=0)

    out = [img_As, img_Bs, label_As, label_Bs]

    cursor = 4

    if sample_len > cursor and torch.is_tensor(first[cursor]):
        changes = torch.stack([item[cursor] for item in batch], dim=0)
        out.append(changes)
        cursor += 1

    if sample_len > cursor and isinstance(first[cursor], dict):
        metas = [item[cursor] for item in batch]
        out.append(metas)
        cursor += 1

    if sample_len > cursor:
        names = [item[cursor] for item in batch]
        out.append(names)

    return tuple(out)


# =========================
# Loader builders
# =========================
def build_train_loader_crcinet(
    batch_size=4,
    num_workers=4,
    pin_memory=True,

    mode='train',
    random_flip=True,
    model_type='ours',

    scale_min=1.0,
    scale_max=4.0,
    use_blur=True,

    p_same=0.30,
    p_A_low=0.35,
    p_B_low=0.35,

    return_change=False,
    return_meta=False,

    drop_last=True,
    shuffle=True,
):
    """
    CRCI-Net 训练 loader。

    特点：
        同一个 batch 内 pair_type 和 scale 一致，
        因此支持 model_type='ours' 下 batch_size > 1。
    """
    train_set = LandSetARDataset(
        mode=mode,
        random_flip=random_flip,
        model_type=model_type,
        scale_min=scale_min,
        scale_max=scale_max,
        use_blur=use_blur,
        p_same=p_same,
        p_A_low=p_A_low,
        p_B_low=p_B_low,
        return_change=return_change,
        return_meta=return_meta,
    )

    batch_sampler = SameScaleBatchSampler(
        dataset=train_set,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        scale_min=scale_min,
        scale_max=scale_max,
        p_same=p_same,
        p_A_low=p_A_low,
        p_B_low=p_B_low,
    )

    loader = DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_stack,
    )

    return loader


def build_eval_loader_crcinet(
    mode='val',
    batch_size=4,
    num_workers=4,
    pin_memory=True,

    model_type='ours',
    eval_pair_type=0,
    eval_scale_A=1.0,
    eval_scale_B=1.0,

    use_blur=True,

    return_change=False,
    return_meta=False,
    visual=False,
):
    """
    固定倍率验证/测试 loader。

    因为整个 eval dataset 使用固定 pair_type / scale_A / scale_B，
    所以 batch 内尺寸天然一致，可以正常 batch_size > 1。
    """
    eval_set = LandSetARDataset(
        mode=mode,
        random_flip=False,
        model_type=model_type,
        use_blur=use_blur,
        eval_pair_type=eval_pair_type,
        eval_scale_A=eval_scale_A,
        eval_scale_B=eval_scale_B,
        return_change=return_change,
        return_meta=return_meta,
        visual=visual,
    )

    loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn_stack,
    )

    return loader


# =========================
# Debug
# =========================
if __name__ == '__main__':
    loader = build_train_loader_crcinet(
        batch_size=4,
        num_workers=0,
        model_type='ours',
        scale_min=1.0,
        scale_max=2.0,
        p_same=0.30,
        p_A_low=0.35,
        p_B_low=0.35,
        return_change=False,
        return_meta=False,
    )

    for i, data in enumerate(loader):
        imgs_A, imgs_B, labels_A, labels_B = data

        print('batch:', i)
        print('imgs_A:', imgs_A.shape)
        print('imgs_B:', imgs_B.shape)
        print('labels_A:', labels_A.shape)
        print('labels_B:', labels_B.shape)

        if i >= 5:
            break