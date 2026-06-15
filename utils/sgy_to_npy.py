# -*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
import random
import create_noisy
import torch
from noise import pnoise2
from add_noise import add_coherent_and_random_noise
from GetPatches import read_segy_data
from scipy.ndimage import gaussian_filter
import os

def add_gaussian_noise(data, mean=0, noise_strength=0.3):
    # 根据噪声强度来计算标准差
    std_dev = noise_strength * np.std(data)  # 使用数据的标准差来缩放噪声
    # 生成与数据形状相同的高斯噪声
    noise = np.random.normal(mean, std_dev, data.shape)
    # 将噪声添加到数据
    noisy_data = data + noise
    return noisy_data

# def add_seismic_noise(data, min_noise_level=0.1, max_noise_level=0.5, smoothness=50):
#     """
#     向未归一化的地震数据添加非平稳高斯噪声。
#     噪声的标准差(Sigma)在空间上平滑变化，最适合验证带有TV Loss的Sigma估计网络。
#
#     参数:
#         data: 原始地震数据 (2D numpy array), 任意幅值范围
#         min_noise_level: 最小噪声强度 (相对于数据全局标准差的倍数)
#         max_noise_level: 最大噪声强度 (相对于数据全局标准差的倍数)
#         smoothness: 噪声强度变化的空间平滑度 (数值越大，Sigma图变化越平缓，越符合TV Loss假设)
#
#     返回:
#         noisy_data: 加噪后的数据
#         real_sigma_map: 真实的噪声标准差图 (Ground Truth Sigma)
#     """
#     h, w = data.shape
#
#     # 1. 获取数据原本的统计特征，用于确定噪声的绝对量级
#     data_std = np.std(data)
#
#     # 2. 生成随机的噪声强度基底
#     # 随机生成一个与原图同大小的图
#     random_base = np.random.rand(h, w)
#
#     # 3. 对强度基底进行高斯模糊，使其变得"平滑"
#     # 这是为了模拟地质环境或采集条件的连续变化，也是为了配合 TV Loss
#     smooth_base = gaussian_filter(random_base, sigma=smoothness)
#
#     # 4. 将平滑后的基底归一化到 [0, 1]
#     smooth_base = (smooth_base - smooth_base.min()) / (smooth_base.max() - smooth_base.min())
#
#     # 5. 将强度图映射到用户指定的物理范围
#     # real_sigma_map 的单位与原始 data 的单位一致
#     min_sigma = min_noise_level * data_std
#     max_sigma = max_noise_level * data_std
#     real_sigma_map = min_sigma + smooth_base * (max_sigma - min_sigma)
#
#     # 6. 生成非平稳高斯噪声
#     # scale 参数接受矩阵，实现逐点不同的标准差
#     noise = np.random.normal(loc=0.0, scale=real_sigma_map, size=(h, w))
#
#     # 7. 叠加噪声
#     noisy_data = data + noise
#
#     return noisy_data
    # return noisy_data, real_sigma_map
import numpy as np
from scipy.ndimage import gaussian_filter

import numpy as np
from scipy.ndimage import gaussian_filter


# def add_seismic_noise_realistic(data, min_noise_level=0.1, max_noise_level=0.5, smoothness=50):
#     """
#     生成更符合地震特征的非平稳噪声：包含平滑背景噪声 + 坏道噪声 + 静音区突变。
#
#     参数:
#     - data: 输入的纯净地震数据 (2D numpy array)
#     - min_noise_level: 最小噪声水平 (相对于数据标准差)
#     - max_noise_level: 最大噪声水平
#     - smoothness: 背景噪声的空间平滑度 (越大越平滑)
#
#     返回:
#     - noisy_data: 加噪后的数据
#     - sigma_map: 真实的噪声水平图 (用于训练 Sigma 估计分支)
#     """
#     h, w = data.shape
#     data_std = np.std(data)
#
#     # ==========================
#     # 1. 基础平滑噪声 - 模拟环境噪声 (Spatially Varying Noise)
#     # ==========================
#     # 生成随机噪声图并进行高斯模糊，模拟噪声的空间连续性
#     random_base = np.random.rand(h, w)
#     smooth_base = gaussian_filter(random_base, sigma=smoothness)
#
#     # 归一化到 0~1
#     if smooth_base.max() != smooth_base.min():
#         smooth_base = (smooth_base - smooth_base.min()) / (smooth_base.max() - smooth_base.min())
#
#     # 映射到 [min, max] 区间
#     sigma_map = min_noise_level * data_std + smooth_base * (max_noise_level - min_noise_level) * data_std
#
#     # ==========================
#     # 2. 添加“坏道” (Bad Traces) - 模拟采集设备故障
#     # ==========================
#     # 这是一个关键步骤：它在 Sigma Map 上创造了剧烈的“垂直条纹”。
#     # 只有让网络见过 Sigma Map 中的垂直条纹，它才能学会预测出图4中的 "Estimated Sigma Map"。
#
#     num_bad_traces = int(w * 0.05)  # 5% 的坏道
#     if num_bad_traces > 0:
#         bad_trace_indices = np.random.choice(w, num_bad_traces, replace=False)
#
#         for idx in bad_trace_indices:
#             # 坏道的噪声水平通常极高 (提升 3-5 倍)
#             boost_factor = np.random.uniform(2.0, 4.0)
#             sigma_map[:, idx] *= boost_factor
#
#     # ==========================
#     # 3. (可选) 恢复“静音区”特征 (Mute Zone) - 但更温和
#     # ==========================
#     # 你之前提到去掉这步导致效果变差。这步的作用是强迫网络处理“大面积高能噪声”。
#     # 这里我们做一个温和版：
#     for col in range(w):
#         mute_boundary = int(col * 0.5)  # 斜率 0.5
#         if mute_boundary < h:
#             # 仅在左上角区域增加额外噪声，倍率随机，模拟初至前的混乱
#             mute_boost = np.random.uniform(1.2, 1.8)
#             sigma_map[0:mute_boundary, col] *= mute_boost
#
#     # ==========================
#     # 4. 生成最终噪声
#     # ==========================
#     # 这一步是物理核心：非平稳噪声 = 标准正态分布 * Sigma Map
#     noise = np.random.normal(loc=0.0, scale=sigma_map, size=(h, w))
#     noisy_data = data + noise
#
#     # return noisy_data, sigma_map
#     return noisy_data
def add_seismic_noise_physics_aware(data, min_noise_level=0.1, max_noise_level=0.5, smoothness=50):
    """
    生成物理一致的非平稳噪声：深度趋势 + 平滑扰动 + 坏道。
    完美支撑论文中 "Gain Correction amplifies deep noise" 的论点。
    """
    h, w = data.shape
    data_std = np.std(data)

    # ==========================================
    # 1. 核心物理特征：随深度增加的噪声趋势 (Key for your Paper!)
    # ==========================================
    # 模拟球面扩散补偿后的效果：深层噪声显著大于浅层
    # 创建一个从 0 到 1 的线性或指数增长的深度权重
    depth_trend = np.linspace(0, 1, h).reshape(h, 1)  # 形状 (h, 1)
    # 扩展到全图
    depth_trend = np.tile(depth_trend, (1, w))

    # ==========================================
    # 2. 局部非平稳性 (Spatially Varying Perturbation)
    # ==========================================
    # 在深度趋势的基础上，增加一些随机的平滑波动，模拟地质环境的不均匀
    random_base = np.random.rand(h, w)
    smooth_fluctuation = gaussian_filter(random_base, sigma=smoothness)

    # 归一化波动项
    if smooth_fluctuation.max() != smooth_fluctuation.min():
        smooth_fluctuation = (smooth_fluctuation - smooth_fluctuation.min()) / \
                             (smooth_fluctuation.max() - smooth_fluctuation.min())

    # === 合成基础 Sigma Map ===
    # 逻辑：基础噪声 + (深度趋势 * 权重) + (随机波动 * 权重)
    # 让深度起主导作用 (0.7), 随机波动起次要作用 (0.3)
    combined_base = 0.7 * depth_trend + 0.3 * smooth_fluctuation

    # 映射到 [min, max] 区间
    sigma_map = min_noise_level * data_std + \
                combined_base * (max_noise_level - min_noise_level) * data_std

    # ==========================================
    # 3. 添加“坏道” (Bad Traces) - TGV 的试金石
    # ==========================================
    # 坏道会在 Sigma Map 上产生剧烈的“垂直条纹”。
    # 你的 SeisPVR 如果能把这种 Sigma 预测出来，说明 TGV 很好的处理了各向异性。
    num_bad_traces = int(w * 0.05)  # 5% 坏道
    if num_bad_traces > 0:
        bad_trace_indices = np.random.choice(w, num_bad_traces, replace=False)
        for idx in bad_trace_indices:
            # 坏道噪声通常是极其剧烈的，且贯穿整个时间轴
            boost_factor = np.random.uniform(2.0, 5.0)
            sigma_map[:, idx] *= boost_factor

    # ==========================================
    # 4. 生成最终噪声
    # ==========================================
    # 异方差高斯噪声：Noise ~ N(0, Sigma_Map)
    noise = np.random.normal(loc=0.0, scale=sigma_map, size=(h, w))
    noisy_data = data + noise

    return noisy_data, sigma_map

def sgy_npy():
    # path = os.path.join('../data/sgy1', 'Anisotropic_FD_Model_Shots_part1.sgy_data_shot')  # 获取sgy数据路径
    # path = '../data/Model94_shots.segy'
    path = '../data/shots0001_0200.segy'
    # path = '../data/sgy_data_gauss/shots0001_0200.segy'
    sgy_data = read_segy_data(path)  # 读取sgy地震数据
    print(sgy_data.shape)   # 查看数据尺寸

    # 数据共30炮，每一炮都是301道地震记录，数据不含噪声，将数据提取为单炮第地震记录
    # for i in range(200):
    #     # 读取干净的炮集
    #     clean_shot = sgy_data[:, i*240:(i+1)*240]
    #     clean_shot = clean_shot.astype(np.float64)
    #     # 保存干净的炮集记录
    #     clean_name = f'noise{i + 1}'  # 给每一个炮集命名，采用format方法
    #     np.save('..\\data\\sgy_data_gauss\\noise\\' + clean_name, clean_shot)  # 设置保存路径

        # 对数据加噪并且保存
        # random_noise_strength = random.choice([0.1, 0.2, 0.3])
        # noise_shot = add_gaussian_noise(clean_shot, noise_strength=random_noise_strength)
        # 保存含噪声的炮集记录
        # noise_name = f'noise{i + 1}'
        # np.save('..\\data\\sgy_data\\noise\\' + noise_name, noise_shot)

    # for i in range(515):
    #     # 读取干净的炮集
    #     clean_shot = sgy_data[:, i*240:(i+1)*240]
    #     # 保存干净的炮集记录
    #     clean_name = f'clean{i + 1}'  # 给每一个炮集命名，采用format方法
    #     np.save('..\\data\\sgy_data\\clean\\' + clean_name, clean_shot)  # 设置保存路径
    #     # print(abs(clean_shot).max())
    #     # 对数据加噪并且保存
    #     # noise_shot = add_seismic_noise_physics_aware(clean_shot)
    #     noise_shot, sigma_map = add_seismic_noise_physics_aware(clean_shot)
    #     # print(abs(noise_shot).max())
    #     sigma_name = f'sigma{i + 1}'
    #     np.save('..\\data\\sgy_data\\sigma\\' + sigma_name, sigma_map)
    #     # 保存含噪声的炮集记录
    #     noise_name = f'noise{i + 1}'
    #     np.save('..\\data\\sgy_data\\noise\\' + noise_name, noise_shot)
    for i in range(5):
        # 读取干净的炮集
        clean_shot = sgy_data[:, i*1201:(i+1)*1201]
        # 保存干净的炮集记录
        clean_name = f'clean{i + 1}'  # 给每一个炮集命名，采用format方法
        np.save('..\\data\\sgy_data_shot\\clean\\' + clean_name, clean_shot)  # 设置保存路径
        # print(abs(clean_shot).max())
        # 对数据加噪并且保存
        # random_noise_strength = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        noise_shot = add_gaussian_noise(clean_shot, noise_strength=0.1)
        # noise_shot, sigma_map = add_seismic_noise_realistic(clean_shot)
        # noise_shot = add_seismic_noise_realistic(clean_shot)
        # print(abs(noise_shot).max())
        # sigma_name = f'sigma{i + 1}'
        # np.save('..\\data\\sgy_data_gauss\\sigma\\' + sigma_name, sigma_map)
        # 保存含噪声的炮集记录
        noise_name = f'noise{i + 1}'
        np.save('..\\data\\sgy_data_shot\\noise\\' + noise_name, noise_shot)

    print(f'第{i + 1}个地震数据已经抽稀保存完毕')

if __name__ == "__main__":
    sgy_npy()
