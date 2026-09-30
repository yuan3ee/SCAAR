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
src_root = '/mnt/sda/su8/SCD/Dataset/Landsat-SCD_dataset/val'        # 原始高分辨率验证源
test_root = '/mnt/sda/su8/SCD/Dataset/Landsat-SCD_dataset/test_CR'   # 用来参考一个子测试集样本数
dst_root = '/mnt/sda/su8/SCD/Dataset/Landsat-SCD_dataset/val_CR'     # 输出固定验证集

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


def find_file_by_stem(folder, stem):
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        p = Path(folder) / f'{stem}{ext}'
        if p.exists():
            return p
    return None


def get_target_total_from_test_subset(test_root, subset='same'):
    subset_A = Path(test_root) / subset / 'A'
    if not subset_A.exists():
        raise FileNotFoundError(f'Cannot find test subset dir: {subset_A}')
    return len(list_valid_names(subset_A))


def save_meta(meta_path, meta_dict):
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)


# =========================
# 主函数
# =========================
def build_fixed_landset_val_mix(
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

    A_dir = Path(src_root) / 'A'
    B_dir = Path(src_root) / 'B'
    label1_dir = Path(src_root) / 'label1'
    label2_dir = Path(src_root) / 'label2'

    if not A_dir.exists():
        raise FileNotFoundError(f'Cannot find {A_dir}')
    if not B_dir.exists():
        raise FileNotFoundError(f'Cannot find {B_dir}')
    if not label1_dir.exists():
        raise FileNotFoundError(f'Cannot find {label1_dir}')
    if not label2_dir.exists():
        raise FileNotFoundError(f'Cannot find {label2_dir}')

    # 输出目录
    out_A = Path(dst_root) / 'A'
    out_B = Path(dst_root) / 'B'
    out_label1 = Path(dst_root) / 'label1'
    out_label2 = Path(dst_root) / 'label2'
    out_meta = Path(dst_root) / 'meta'

    for d in [out_A, out_B, out_label1, out_label2, out_meta]:
        ensure_dir(d)

    # 目标总量 = 一个子测试集样本数
    target_total = get_target_total_from_test_subset(test_root, subset='same')

    # 按 50/25/25 划分
    n_same = int(target_total * ratio_same)
    n_A_low = int(target_total * ratio_A_low)
    n_B_low = target_total - n_same - n_A_low

    names = list_valid_names(A_dir)
    if len(names) < target_total:
        raise ValueError(f'源验证集样本数不够: have {len(names)}, need {target_total}')

    random.shuffle(names)
    selected = names[:target_total]

    same_names = selected[:n_same]
    A_low_names = selected[n_same:n_same + n_A_low]
    B_low_names = selected[n_same + n_A_low:]

    def process_one(name, subset_type):
        stem = Path(name).stem

        img_A_path = A_dir / name
        img_B_path = find_file_by_stem(B_dir, stem)
        label1_path = find_file_by_stem(label1_dir, stem)
        label2_path = find_file_by_stem(label2_dir, stem)

        if img_B_path is None or label1_path is None or label2_path is None:
            raise FileNotFoundError(f'Cannot find matched file for {name}')

        img_A = io.imread(img_A_path)
        img_B = io.imread(img_B_path)
        label_A = io.imread(label1_path)
        label_B = io.imread(label2_path)

        if subset_type == 'same':
            out_img_A = img_A
            out_img_B = img_B
            pair_type = 'same'
            scale_A = 1.0
            scale_B = 1.0

        elif subset_type == 'A_low':
            out_img_A = degrade_resolution(img_A, scale=scale, use_blur=True)
            out_img_B = img_B
            pair_type = 'A_low'
            scale_A = float(scale)
            scale_B = 1.0

        elif subset_type == 'B_low':
            out_img_A = img_A
            out_img_B = degrade_resolution(img_B, scale=scale, use_blur=True)
            pair_type = 'B_low'
            scale_A = 1.0
            scale_B = float(scale)

        else:
            raise ValueError(subset_type)

        io.imsave(out_A / img_A_path.name, out_img_A, check_contrast=False)
        io.imsave(out_B / img_B_path.name, out_img_B, check_contrast=False)
        io.imsave(out_label1 / label1_path.name, label_A, check_contrast=False)
        io.imsave(out_label2 / label2_path.name, label_B, check_contrast=False)

        meta = {
            'name': img_A_path.name,
            'pair_type': pair_type,
            'scale_A': scale_A,
            'scale_B': scale_B,
            'input_size_A': list(out_img_A.shape[:2]),
            'input_size_B': list(out_img_B.shape[:2]),
            'target_size': list(label_A.shape[:2]),
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
    build_fixed_landset_val_mix(
        src_root=src_root,
        dst_root=dst_root,
        test_root=test_root,
        ratio_same=0.50,
        ratio_A_low=0.25,
        ratio_B_low=0.25,
        scale=2.0,
        seed=seed,
    )