import os
import json
import random
from pathlib import Path

import cv2
import numpy as np
from skimage import io
from tqdm import tqdm


# =========================
# 路径
# =========================
src_root = '/mnt/sda/su8/SCD/Dataset/JL1_second/val'         # 原始高分辨率验证源
test_root = '/mnt/sda/su8/SCD/Dataset/JL1_second/test_CR'    # 用来参考一个子测试集样本数
dst_root = '/mnt/sda/su8/SCD/Dataset/JL1_second/val_CR'      # 输出固定验证集

seed = 2026


# =========================
# 工具函数
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def degrade_resolution(img, scale=2.0, use_blur=True):
    """
    真实降分辨率，不上采样回原尺寸
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
    return img_low


def list_valid_names(folder):
    valid_ext = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    return sorted([p.name for p in Path(folder).iterdir() if p.suffix.lower() in valid_ext])


def find_label_path(label_dir, stem):
    """
    JL1 标签通常是 png，兼容检索
    """
    for ext in ['.png', '.tif', '.tiff']:
        p = Path(label_dir) / f'{stem}{ext}'
        if p.exists():
            return p
    return None


def get_target_total_from_test_subset(test_root, subset='same'):
    subset_im1 = Path(test_root) / subset / 'im1'
    if not subset_im1.exists():
        raise FileNotFoundError(f'Cannot find test subset dir: {subset_im1}')
    return len(list_valid_names(subset_im1))


def save_meta(meta_path, meta_dict):
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)


# =========================
# 主函数
# =========================
def build_fixed_jl1_val_mix(
    src_root,
    dst_root,
    test_root,
    ratio_same=0.50,
    ratio_A_low=0.25,
    ratio_B_low=0.25,
    scale=2.0,
    seed=2026,
):
    random.seed(seed)
    np.random.seed(seed)

    img_A_dir = Path(src_root) / 'im1'
    img_B_dir = Path(src_root) / 'im2'
    label_A_dir = Path(src_root) / 'label1_gray'
    label_B_dir = Path(src_root) / 'label2_gray'

    if not img_A_dir.exists():
        raise FileNotFoundError(f'Cannot find {img_A_dir}')
    if not img_B_dir.exists():
        raise FileNotFoundError(f'Cannot find {img_B_dir}')
    if not label_A_dir.exists():
        raise FileNotFoundError(f'Cannot find {label_A_dir}')
    if not label_B_dir.exists():
        raise FileNotFoundError(f'Cannot find {label_B_dir}')

    # 输出目录
    out_im1 = Path(dst_root) / 'im1'
    out_im2 = Path(dst_root) / 'im2'
    out_label1 = Path(dst_root) / 'label1_gray'
    out_label2 = Path(dst_root) / 'label2_gray'
    out_meta = Path(dst_root) / 'meta'

    for d in [out_im1, out_im2, out_label1, out_label2, out_meta]:
        ensure_dir(d)

    # 目标总量 = 一个子测试集样本数
    target_total = get_target_total_from_test_subset(test_root, subset='same')

    # 按 50/25/25 划分
    n_same = int(target_total * ratio_same)
    n_A_low = int(target_total * ratio_A_low)
    n_B_low = target_total - n_same - n_A_low

    names = list_valid_names(img_A_dir)
    if len(names) < target_total:
        raise ValueError(f'源验证集样本数不够: have {len(names)}, need {target_total}')

    random.shuffle(names)
    selected = names[:target_total]

    same_names = selected[:n_same]
    A_low_names = selected[n_same:n_same + n_A_low]
    B_low_names = selected[n_same + n_A_low:]

    def process_one(name, subset_type):
        stem = Path(name).stem

        img_A_path = img_A_dir / name
        img_B_path = img_B_dir / name

        label_A_path = find_label_path(label_A_dir, stem)
        label_B_path = find_label_path(label_B_dir, stem)

        if label_A_path is None or label_B_path is None:
            raise FileNotFoundError(f'Cannot find labels for {name}')

        img_A = io.imread(img_A_path)
        img_B = io.imread(img_B_path)
        label_A = io.imread(label_A_path)
        label_B = io.imread(label_B_path)

        if subset_type == 'same':
            out_A = img_A
            out_B = img_B
            pair_type = 'same'
            scale_A = 1.0
            scale_B = 1.0

        elif subset_type == 'A_low':
            out_A = degrade_resolution(img_A, scale=scale, use_blur=True)
            out_B = img_B
            pair_type = 'A_low'
            scale_A = float(scale)
            scale_B = 1.0

        elif subset_type == 'B_low':
            out_A = img_A
            out_B = degrade_resolution(img_B, scale=scale, use_blur=True)
            pair_type = 'B_low'
            scale_A = 1.0
            scale_B = float(scale)

        else:
            raise ValueError(subset_type)

        io.imsave(out_im1 / name, out_A, check_contrast=False)
        io.imsave(out_im2 / name, out_B, check_contrast=False)
        io.imsave(out_label1 / label_A_path.name, label_A, check_contrast=False)
        io.imsave(out_label2 / label_B_path.name, label_B, check_contrast=False)

        meta = {
            'name': name,
            'pair_type': pair_type,
            'scale_A': scale_A,
            'scale_B': scale_B,
            'input_size_A': list(out_A.shape[:2]),
            'input_size_B': list(out_B.shape[:2]),
            'target_size': list(label_A.shape[:2]),   # JL1 通常 256x256
        }
        save_meta(out_meta / f'{stem}.json', meta)

    print(f'target_total = {target_total}')
    print(f'same = {n_same}, A_low_x{scale} = {n_A_low}, B_low_x{scale} = {n_B_low}')

    for name in tqdm(same_names, desc='same'):
        process_one(name, 'same')

    for name in tqdm(A_low_names, desc=f'A_low_x{scale}'):
        process_one(name, 'A_low')

    for name in tqdm(B_low_names, desc=f'B_low_x{scale}'):
        process_one(name, 'B_low')

    split_info = {
        'target_total': target_total,
        'same': n_same,
        'A_low': n_A_low,
        'B_low': n_B_low,
        'scale': scale,
        'seed': seed,
    }
    save_meta(Path(dst_root) / 'split_info.json', split_info)
    print(f'Finished. Saved to: {dst_root}')


if __name__ == '__main__':
    build_fixed_jl1_val_mix(
        src_root=src_root,
        dst_root=dst_root,
        test_root=test_root,
        ratio_same=0.50,
        ratio_A_low=0.25,
        ratio_B_low=0.25,
        scale=2.0,
        seed=seed,
    )