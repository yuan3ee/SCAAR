import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from skimage import io
import cv2


# =========================
# JL1
# =========================
num_classes = 6

ST_COLORMAP = [
    [255, 255, 255],   # unchanged
    [0, 128, 0],       # farm
    [0, 0, 128],       # road
    [128, 0, 0],       # tree
    [0, 128, 128],     # building
    [128, 128, 0],     # other
]
ST_CLASSES = ['unchanged', 'farm', 'road', 'tree', 'building', 'other']

MEAN_A = np.array([113.40, 114.08, 116.45], dtype=np.float32)
STD_A  = np.array([48.30,  46.27,  48.14], dtype=np.float32)
MEAN_B = np.array([111.07, 114.04, 118.18], dtype=np.float32)
STD_B  = np.array([49.41,  47.01,  47.94], dtype=np.float32)

root = '/mnt/sda/su8/SCD/Dataset/JL1_second'


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


def find_label_path(label_dir, stem):
    for ext in ['.png', '.tif', '.tiff']:
        p = os.path.join(label_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def read_RSimages_paths(mode):
    img_A_dir = os.path.join(root, mode, 'im1')
    img_B_dir = os.path.join(root, mode, 'im2')
    label_A_dir = os.path.join(root, mode, 'label1_gray')
    label_B_dir = os.path.join(root, mode, 'label2_gray')

    data_list = sorted(os.listdir(img_A_dir))

    imgs_list_A, imgs_list_B, labels_A, labels_B = [], [], [], []
    count = 0
    for it in data_list:
        if it.endswith('.png') or it.endswith('.tif') or it.endswith('.tiff'):
            stem = Path(it).stem

            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)

            label_A_path = find_label_path(label_A_dir, stem)
            label_B_path = find_label_path(label_B_dir, stem)

            if label_A_path is None or label_B_path is None:
                raise FileNotFoundError(f'Cannot find label for {it}')

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
        img, (ksize, ksize),
        sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101
    )


def degrade_resolution(img, scale, keep_input_size=False, use_blur=True):
    """
    分辨率退化：
    - 先模糊
    - 再下采样
    - 若 keep_input_size=True，则再插值回原始尺寸
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

    img_low = cv2.resize(img, (low_w, low_h), interpolation=cv2.INTER_AREA)

    if keep_input_size:
        img_out = cv2.resize(img_low, (w, h), interpolation=cv2.INTER_CUBIC)
        return img_out
    else:
        return img_low


# =========================
# 数据集
# =========================
class JL1ARDataset(Dataset):
    """
    Resolution-arbitrary JL1 dataset

    model_type:
        - 'ours'     : 输入可以不同尺寸
        - 'baseline' : 退化后再上采样到原始尺寸

    pair_type:
        0: (1,1)
        1: (s,1)  -> A degraded
        2: (1,s)  -> B degraded
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
            raise ValueError(f'p_same + p_A_low + p_B_low must be 1.0, got {prob_sum}')

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
        u = np.random.rand()
        if u < self.p_same:
            pair_type = 0
            scale_A, scale_B = 1.0, 1.0
        elif u < self.p_same + self.p_A_low:
            pair_type = 1
            scale_A = sample_scale_log_uniform(self.scale_min, self.scale_max)
            scale_B = 1.0
        else:
            pair_type = 2
            scale_A = 1.0
            scale_B = sample_scale_log_uniform(self.scale_min, self.scale_max)

        return pair_type, scale_A, scale_B

    def _get_eval_pair(self):
        return self.eval_pair_type, self.eval_scale_A, self.eval_scale_B

    def __getitem__(self, idx):
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

        if self.random_flip:
            img_A, img_B, label_A, label_B = rand_rot90_flip_MCD(
                img_A, img_B, label_A, label_B
            )

        if self.mode == 'train':
            pair_type, scale_A, scale_B = self._sample_train_pair()
        else:
            pair_type, scale_A, scale_B = self._get_eval_pair()

        if self.model_type == 'ours':
            img_A = degrade_resolution(
                img_A, scale_A, keep_input_size=False, use_blur=self.use_blur
            )
            img_B = degrade_resolution(
                img_B, scale_B, keep_input_size=False, use_blur=self.use_blur
            )
        else:
            img_A = degrade_resolution(
                img_A, scale_A, keep_input_size=True, use_blur=self.use_blur
            )
            img_B = degrade_resolution(
                img_B, scale_B, keep_input_size=True, use_blur=self.use_blur
            )

        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')

        label_A = label_A.astype(np.int64)
        label_B = label_B.astype(np.int64)
        change = (label_A != label_B).astype(np.int64)

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
            'name': self.get_mask_name(idx)
        }

        if self.visual:
            out = [img_A_tensor, img_B_tensor, label_A_tensor, label_B_tensor]
            if self.return_change:
                out.append(change_tensor)
            if self.return_meta:
                out.append(meta)
            return tuple(out)

        out = [img_A_tensor, img_B_tensor, label_A_tensor, label_B_tensor]
        if self.return_change:
            out.append(change_tensor)
        if self.return_meta:
            out.append(meta)

        return tuple(out)


# =========================
# collate_fn
# =========================
def collate_fn_ours(batch):
    """
    用于你的方法：A/B 可能不同尺寸，不能直接 stack
    """
    img_As, img_Bs, label_As, label_Bs, changes, metas = zip(*batch)

    label_As = torch.stack(label_As, dim=0)
    label_Bs = torch.stack(label_Bs, dim=0)
    changes = torch.stack(changes, dim=0)

    return list(img_As), list(img_Bs), label_As, label_Bs, changes, list(metas)


def collate_fn_baseline(batch):
    """
    用于对比方法：A/B 都是原始尺寸，可直接 stack
    """
    img_As, img_Bs, label_As, label_Bs, changes, metas = zip(*batch)

    img_As = torch.stack(img_As, dim=0)
    img_Bs = torch.stack(img_Bs, dim=0)
    label_As = torch.stack(label_As, dim=0)
    label_Bs = torch.stack(label_Bs, dim=0)
    changes = torch.stack(changes, dim=0)

    return img_As, img_Bs, label_As, label_Bs, changes, list(metas)


# =========================
# 示例
# =========================
train_set_ours = JL1ARDataset(
    mode='train',
    random_flip=True,
    model_type='ours',
    scale_min=1.0,
    scale_max=4.0,
    p_same=0.30,
    p_A_low=0.35,
    p_B_low=0.35,
    return_change=False,
    return_meta=False,
)

train_set_base = JL1ARDataset(
    mode='train',
    random_flip=True,
    model_type='baseline',
    scale_min=1.0,
    scale_max=4.0,
    p_same=0.30,
    p_A_low=0.35,
    p_B_low=0.35,
    return_change=False,
    return_meta=False,
)