# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
此脚本用于使用DDP（分布式数据并行）从预训练的DiT模型采样大量图像
主要功能：生成大量图像样本并保存为.npz文件，该文件可用于计算FID和其他评估指标
可通过ADM仓库使用：https://github.com/openai/guided-diffusion/tree/main/evaluations

如需简单的单GPU/CPU采样脚本，请参见sample.py。
"""
import torch
import torch.distributed as dist
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    从包含.png样本的文件夹构建单个.npz文件。
    """
    samples = []
    for i in tqdm(range(num), desc="正在从样本构建.npz文件"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"已将.npz文件保存到 {npz_path} [形状={samples.shape}]。")
    return npz_path


def main(args):
    """
    运行采样过程。
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True：速度快但可能导致一些小的数值差异
    assert torch.cuda.is_available(), "使用DDP采样至少需要一个GPU。sample.py支持仅使用CPU"
    torch.set_grad_enabled(False)

    # 设置DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"启动 rank={rank}, seed={seed}, world_size={dist.get_world_size()}。")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "只有DiT-XL/2模型可用于自动下载。"
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # 加载模型:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # 自动下载预训练模型或从train.py加载自定义DiT检查点:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # 重要！
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "在几乎所有情况下，cfg_scale应该 >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # 创建保存样本的文件夹:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"正在将.png样本保存到 {sample_folder_dir}")
    dist.barrier()

    # 计算每个GPU需要生成多少样本以及需要运行多少次迭代:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # 为了使样本数均匀可分，我们会采样比需要的稍多一些，然后丢弃多余的样本:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"将要采样的图像总数: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples必须能被world_size整除"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu必须能被每个GPU的批量大小整除"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # 样本输入:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # 设置无分类器引导:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # 采样图像:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # 移除空类别样本

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # 将样本作为单独的.png文件保存到磁盘
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # 确保所有进程都已完成保存样本，然后尝试转换为.npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("完成。")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="默认情况下，使用TF32矩阵乘法。这在Ampere GPU上极大地加速了采样。")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="可选的DiT检查点路径（默认：自动下载预训练的DiT-XL/2模型）。")
    args = parser.parse_args()
    main(args)
