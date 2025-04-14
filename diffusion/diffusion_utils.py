# 文件功能：扩散模型的实用工具函数
# 本文件提供了扩散模型所需的各种数学工具函数，包括KL散度计算、
# 高斯分布的概率密度函数计算等。这些函数是扩散模型理论基础的核心组件。
#
# 修改自OpenAI的扩散模型代码库
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import torch as th
import numpy as np


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    计算两个高斯分布之间的KL散度。
    形状可以自动广播，因此批次可以与标量进行比较，
    以及其他用例。
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "至少有一个参数必须是张量"

    # 强制方差为张量。广播有助于将标量转换为张量，
    # 但对于th.exp()不起作用。
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    标准正态分布累积分布函数的快速近似。
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def continuous_gaussian_log_likelihood(x, *, means, log_scales):
    """
    计算连续高斯分布的对数似然。
    :param x: 目标
    :param means: 高斯均值张量。
    :param log_scales: 高斯对数标准差张量。
    :return: 与x形状相同的对数概率张量（以奈特为单位）。
    """
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    normalized_x = centered_x * inv_stdv
    log_probs = th.distributions.Normal(th.zeros_like(x), th.ones_like(x)).log_prob(normalized_x)
    return log_probs


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    计算离散化到给定图像的高斯分布的对数似然。
    :param x: 目标图像。假设这是uint8值，重新缩放到[-1, 1]范围。
    :param means: 高斯均值张量。
    :param log_scales: 高斯对数标准差张量。
    :return: 与x形状相同的对数概率张量（以奈特为单位）。
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
