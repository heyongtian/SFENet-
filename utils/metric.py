import math
from warnings import warn

import numpy as np
import torch
from skimage._shared.utils import check_shape_equality
# from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics.simple_metrics import _as_floats, mean_squared_error
from skimage.util.dtype import dtype_range
import cv2
from scipy.ndimage import convolve

def compare_SNR(real_img, recov_img):
    """
    计算信噪比
    :param recov_img: 重建后或者含有噪声的数据
    :param real_img: 干净的数据
    :return: 信噪比
    """
    real_mean = np.mean(real_img)
    tmp1 = real_img - real_mean
    real_var = np.sum(tmp1 * tmp1)  # 计算真实数据的方差

    noise = real_img - recov_img
    noise_mean = np.mean(noise)
    tmp2 = noise - noise_mean
    noise_var = np.sum(tmp2 * tmp2)  # 计算噪声数据的方差

    if noise_var == 0 or real_var == 0:
        s = 999.99  # 如果噪声或真实数据的方差为0，返回一个异常的信噪比
    else:
        # 使用 np.log 计算以 10 为底的对数
        s = 10 * np.log10(real_var / noise_var)  # np.log10 直接计算以 10 为底的对数
    return s

def torch2numpy(tensor, gamma=None):
    # tensor = torch.clamp(tensor, 0.0, 1.0)
    # Convert to 0 - 255
    if gamma is not None:
        tensor = torch.pow(tensor, gamma)
    # tensor = tensor*255
    while len(tensor.size()) < 4:
        tensor = tensor.unsqueeze(1)
    # return tensor.permute(0, 2, 3, 1).cpu().data.numpy()

    tensor_numpy = tensor.permute(0, 2, 3, 1).cpu().data.numpy()
    return tensor_numpy.astype(np.float64)

#FSIM
# 计算梯度幅值
def gradient_magnitude(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    return grad

# 简化的相位一致性近似，用拉普拉斯滤波器
def phase_congruency(img):
    kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    pc = convolve(img, kernel, mode='reflect')
    pc = np.abs(pc)
    return pc / (pc.max() + 1e-12)

# 单张图像FSIM计算
def fsim_single(img_ref, img_dist):
    PC1 = phase_congruency(img_ref)
    PC2 = phase_congruency(img_dist)

    GM1 = gradient_magnitude(img_ref)
    GM2 = gradient_magnitude(img_dist)

    T1 = 0.85
    T2 = 160

    PC_sim = (2 * PC1 * PC2 + T1) / (PC1**2 + PC2**2 + T1)
    GM_sim = (2 * GM1 * GM2 + T2) / (GM1**2 + GM2**2 + T2)

    FSIM_map = PC_sim * GM_sim
    PC_max = np.maximum(PC1, PC2)
    fsim_val = np.sum(FSIM_map * PC_max) / (np.sum(PC_max) + 1e-12)
    return fsim_val

# 批量FSIM计算，兼容你的tensor输入格式
def calculate_fsim(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    target_tf = target_tf[..., 0]
    output_tf = output_tf[..., 0]

    fsim = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        fsim += fsim_single(target_tf[im_idx, ...], output_tf[im_idx, ...])
        n += 1.0
    return fsim / n

def calculate_snr(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    target_tf = target_tf[..., 0]
    output_tf = output_tf[..., 0]
    snr = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        snr += compare_SNR(target_tf[im_idx, ...],
                             output_tf[im_idx, ...])
        n += 1.0
    return snr / n

def calculate_psnr(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    target_tf = target_tf[..., 0]
    output_tf = output_tf[..., 0]
    psnr = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        # psnr += compare_psnr(target_tf[im_idx, ...],
        #                                      output_tf[im_idx, ...], data_range=2.0)
        psnr += compare_psnr(target_tf[im_idx, ...],
                             output_tf[im_idx, ...], data_range=2.0)
        n += 1.0
    return psnr / n

def calculate_rmse(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    target_tf = target_tf[..., 0]
    output_tf = output_tf[..., 0]
    rmse = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        rmse += np.sqrt(mean_squared_error(target_tf[im_idx, ...],
                                             output_tf[im_idx, ...]))
        n += 1.0
    return rmse / n

def calculate_ssim(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    target_tf = target_tf[..., 0]
    output_tf = output_tf[..., 0]
    ssim = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        ssim += compare_ssim(target_tf[im_idx, ...],
                                             output_tf[im_idx, ...],
                                             multichannel=True,
                                             data_range=2.0)
        n += 1.0
    return ssim / n

