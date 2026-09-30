import os
import json
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from skimage import io


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


def degrade_resolution(img, scale, use_blur=True):
    """
    真实降分辨率，不上采样回原尺寸
    输入: H,W,3
    输出: H/scale, W/scale, 3
    """
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


def save_meta(meta_path, meta_dict):
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)


def build_second_ar_testset(
    src_root,
    dst_root,
    scales=(1.5, 2.0, 3.0, 4.0),
    use_blur=True,
):
    """
    基于 SECOND 原始 test_CR 离线构造任意分辨率测试集
    只做:
        same
        A_low_xs
        B_low_xs
    """

    src_root = Path(src_root)
    dst_root = Path(dst_root)

    im1_dir = src_root / 'im1'
    im2_dir = src_root / 'im2'
    label1_dir = src_root / 'label1'
    label2_dir = src_root / 'label2'

    file_list = sorted([p.name for p in im1_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']])

    # -------- 1. same --------
    same_dir = dst_root / 'same'
    for sub in ['im1', 'im2', 'label1', 'label2', 'meta']:
        ensure_dir(same_dir / sub)

    print('Building same ...')
    for name in tqdm(file_list):
        img_A = io.imread(im1_dir / name)
        img_B = io.imread(im2_dir / name)
        label_A = io.imread(label1_dir / name)
        label_B = io.imread(label2_dir / name)

        io.imsave(same_dir / 'im1' / name, img_A, check_contrast=False)
        io.imsave(same_dir / 'im2' / name, img_B, check_contrast=False)
        io.imsave(same_dir / 'label1' / name, label_A, check_contrast=False)
        io.imsave(same_dir / 'label2' / name, label_B, check_contrast=False)

        meta = {
            'name': name,
            'pair_type': 'same',
            'scale_A': 1.0,
            'scale_B': 1.0,
            'input_size_A': list(img_A.shape[:2]),
            'input_size_B': list(img_B.shape[:2]),
            'target_size': list(label_A.shape[:2]),
        }
        save_meta(same_dir / 'meta' / f'{Path(name).stem}.json', meta)

    # -------- 2. A_low_xs / B_low_xs --------
    for s in scales:
        # A_low_xs
        a_dir = dst_root / f'A_low_x{s}'
        for sub in ['im1', 'im2', 'label1', 'label2', 'meta']:
            ensure_dir(a_dir / sub)

        print(f'Building A_low_x{s} ...')
        for name in tqdm(file_list):
            img_A = io.imread(im1_dir / name)
            img_B = io.imread(im2_dir / name)
            label_A = io.imread(label1_dir / name)
            label_B = io.imread(label2_dir / name)

            img_A_low = degrade_resolution(img_A, s, use_blur=use_blur)

            io.imsave(a_dir / 'im1' / name, img_A_low, check_contrast=False)
            io.imsave(a_dir / 'im2' / name, img_B, check_contrast=False)
            io.imsave(a_dir / 'label1' / name, label_A, check_contrast=False)
            io.imsave(a_dir / 'label2' / name, label_B, check_contrast=False)

            meta = {
                'name': name,
                'pair_type': 'A_low',
                'scale_A': float(s),
                'scale_B': 1.0,
                'input_size_A': list(img_A_low.shape[:2]),
                'input_size_B': list(img_B.shape[:2]),
                'target_size': list(label_A.shape[:2]),
            }
            save_meta(a_dir / 'meta' / f'{Path(name).stem}.json', meta)

        # B_low_xs
        b_dir = dst_root / f'B_low_x{s}'
        for sub in ['im1', 'im2', 'label1', 'label2', 'meta']:
            ensure_dir(b_dir / sub)

        print(f'Building B_low_x{s} ...')
        for name in tqdm(file_list):
            img_A = io.imread(im1_dir / name)
            img_B = io.imread(im2_dir / name)
            label_A = io.imread(label1_dir / name)
            label_B = io.imread(label2_dir / name)

            img_B_low = degrade_resolution(img_B, s, use_blur=use_blur)

            io.imsave(b_dir / 'im1' / name, img_A, check_contrast=False)
            io.imsave(b_dir / 'im2' / name, img_B_low, check_contrast=False)
            io.imsave(b_dir / 'label1' / name, label_A, check_contrast=False)
            io.imsave(b_dir / 'label2' / name, label_B, check_contrast=False)

            meta = {
                'name': name,
                'pair_type': 'B_low',
                'scale_A': 1.0,
                'scale_B': float(s),
                'input_size_A': list(img_A.shape[:2]),
                'input_size_B': list(img_B_low.shape[:2]),
                'target_size': list(label_A.shape[:2]),
            }
            save_meta(b_dir / 'meta' / f'{Path(name).stem}.json', meta)

    print(f'Finished! Saved to: {dst_root}')


if __name__ == '__main__':
    src_root = '/mnt/sda/su8/SCD/Dataset/second/test'          # 你当前原始测试集
    dst_root = '/mnt/sda/su8/SCD/Dataset/second/test_CR'       # 新生成的离线测试集目录

    build_second_ar_testset(
        src_root=src_root,
        dst_root=dst_root,
        scales=(2.5, 3.5, 4.0, 4.5, 5.0),
        use_blur=True,
    )