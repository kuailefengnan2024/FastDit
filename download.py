# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
此脚本用于下载预训练的DiT模型
主要功能：提供自动下载预训练DiT模型的功能，或者加载本地已有的模型
"""
from torchvision.datasets.utils import download_url
import torch
import os


pretrained_models = {'DiT-XL-2-512x512.pt', 'DiT-XL-2-256x256.pt'}


def find_model(model_name):
    """
    查找预训练的DiT模型，如果必要的话会自动下载。或者，从本地路径加载模型。
    """
    if model_name in pretrained_models:  # 查找/下载我们预训练的DiT检查点
        return download_model(model_name)
    else:  # 加载自定义DiT检查点:
        assert os.path.isfile(model_name), f'无法在{model_name}找到DiT检查点'
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # 支持来自train.py的检查点
            checkpoint = checkpoint["ema"]
        return checkpoint


def download_model(model_name):
    """
    从网络下载预训练的DiT模型。
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://dl.fbaipublicfiles.com/DiT/models/{model_name}'
        download_url(web_path, 'pretrained_models')
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


if __name__ == "__main__":
    # 下载所有DiT检查点
    for model in pretrained_models:
        download_model(model)
    print('完成。')
