# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
此脚本用于训练DiT（扩散Transformer）模型
主要功能：实现了DiT模型的完整训练流程，包括数据加载、模型初始化、训练循环和检查点保存
"""
import torch
# 当我们测试这个脚本时，下面的第一个标志是False，但在A100上设为True可以使训练速度更快:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             训练辅助函数                                       #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    将EMA模型向当前模型迭代更新。
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: 考虑只对需要梯度的参数应用，以避免pos_embed的小数值变化
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    为模型中的所有参数设置requires_grad标志。
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    创建一个记录到日志文件和标准输出的记录器。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    来自ADM的中心裁剪实现。
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "特征文件和标签文件的数量应该相同"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


#################################################################################
#                                  训练循环                                      #
#################################################################################

def main(args):
    """
    训练新的DiT模型。
    """
    assert torch.cuda.is_available(), "训练当前至少需要一个GPU。"

    # 设置加速器:
    accelerator = Accelerator()
    device = accelerator.device

    # 设置实验文件夹:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # 创建结果文件夹（包含所有实验子文件夹）
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # 例如，DiT-XL/2 --> DiT-XL-2（用于命名文件夹）
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # 创建实验文件夹
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # 存储保存的模型检查点
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"实验目录创建于 {experiment_dir}")

    # 创建模型:
    assert args.image_size % 8 == 0, "图像大小必须能被8整除（用于VAE编码器）。"
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # 注意，参数初始化是在DiT构造函数内完成的
    model = model.to(device)
    ema = deepcopy(model).to(device)  # 创建模型的EMA副本，用于训练后使用
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # 默认：1000步，线性噪声调度
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    if accelerator.is_main_process:
        logger.info(f"DiT参数: {sum(p.numel() for p in model.parameters()):,}")

    # 设置优化器（在我们的论文中，我们使用了默认的Adam参数beta=(0.9, 0.999)和1e-4的恒定学习率）:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # 设置数据:
    features_dir = f"{args.feature_path}/imagenet256_features"
    labels_dir = f"{args.feature_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"数据集包含 {len(dataset):,} 张图像 ({args.feature_path})")

    # 准备模型进行训练:
    update_ema(ema, model, decay=0)  # 确保EMA使用同步的权重进行初始化
    model.train()  # 重要！这启用了用于无分类器引导的嵌入丢弃
    ema.eval()  # EMA模型应始终处于评估模式
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # 用于监控/记录的变量:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"训练 {args.epochs} 个周期...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"开始第 {epoch} 个周期...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # 记录损失值:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # 测量训练速度:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # 在所有进程上减少损失历史:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(f"(步骤={train_steps:07d}) 训练损失: {avg_loss:.4f}, 训练步数/秒: {steps_per_sec:.2f}")
                # 重置监控变量:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # 保存DiT检查点:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"已将检查点保存到 {checkpoint_path}")

    model.eval()  # 重要！这禁用了随机嵌入丢弃
    # 在评估模式下使用ema（或model）进行任何采样/FID计算/等...
    
    if accelerator.is_main_process:
        logger.info("完成！")


if __name__ == "__main__":
    # 这里的默认参数将训练DiT-XL/2，使用了我们论文中的超参数（除了训练迭代次数）。
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # 选择不影响训练
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
