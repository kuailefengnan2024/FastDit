## 基于 Transformer 的可扩展扩散模型 (DiT)<br><sub>改进的 PyTorch 实现</sub>

### [论文](http://arxiv.org/abs/2212.09748) | [项目页面](https://www.wpeebles.com/DiT) | 运行 DiT-XL/2 [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wpeebles/DiT) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/facebookresearch/DiT/blob/main/run_DiT.ipynb) <a href="https://replicate.com/arielreplicate/scalable_diffusion_with_transformers"><img src="https://replicate.com/arielreplicate/scalable_diffusion_with_transformers/badge"></a>

![DiT 样本](visuals/sample_grid_0.png)

本仓库提供了论文 [**Scalable Diffusion Models with Transformers**](https://www.wpeebles.com/DiT) 的改进 PyTorch 实现。

它包含：

* 🪐 改进的 PyTorch [实现](models.py) 和原始 [实现](train_options/models_original.py) 的 DiT
* ⚡️ 在 ImageNet 上训练的预训练类条件 DiT 模型 (512x512 和 256x256)
* 💥 用于运行预训练 DiT-XL/2 模型的独立 [Hugging Face Space](https://huggingface.co/spaces/wpeebles/DiT) 和 [Colab 笔记本](http://colab.research.google.com/github/facebookresearch/DiT/blob/main/run_DiT.ipynb)
* 🛸 改进的 DiT [训练脚本](train.py) 和多种 [训练选项](train_options)

## 设置

首先，下载并设置仓库：

```bash
git clone https://github.com/chuanyangjin/fast-DiT.git
cd DiT
```

我们提供了一个 [`environment.yml`](environment.yml) 文件，可用于创建 Conda 环境。如果您只想在 CPU 上本地运行预训练模型，可以从文件中删除 `cudatoolkit` 和 `pytorch-cuda` 要求。

```bash
conda env create -f environment.yml
conda activate DiT
```


## 采样 [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wpeebles/DiT) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/facebookresearch/DiT/blob/main/run_DiT.ipynb)
![更多 DiT 样本](visuals/sample_grid_1.png)

**预训练 DiT 检查点。** 您可以使用 [`sample.py`](sample.py) 从我们的预训练 DiT 模型中采样。根据您使用的模型，预训练 DiT 模型的权重将自动下载。该脚本有各种参数，可以在 256x256 和 512x512 模型之间切换，调整采样步骤，更改无分类器引导尺度等。例如，要从我们的 512x512 DiT-XL/2 模型中采样，您可以使用：

```bash
python sample.py --image-size 512 --seed 1
```

为了方便起见，您也可以直接在这里下载我们的预训练 DiT 模型：

| DiT 模型     | 图像分辨率 | FID-50K | Inception Score | Gflops | 
|---------------|------------------|---------|-----------------|--------|
| [XL/2](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) | 256x256          | 2.27    | 278.24          | 119    |
| [XL/2](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt) | 512x512          | 3.04    | 240.82          | 525    |


**自定义 DiT 检查点。** 如果您使用 [`train.py`](train.py) 训练了新的 DiT 模型（参见[下文](#训练-dit)），您可以添加 `--ckpt` 参数来使用您自己的检查点。例如，要从自定义 256x256 DiT-L/4 模型的 EMA 权重中采样，请运行：

```bash
python sample.py --model DiT-L/4 --image-size 256 --ckpt /path/to/model.pt
```


## 训练
### 训练前准备
使用 `1` 个 GPU 在一个节点上提取 ImageNet 特征：

```bash
torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-XL/2 --data-path /path/to/imagenet/train --features-path /path/to/store/features
```

### 训练 DiT
我们在 [`train.py`](train.py) 中提供了 DiT 的训练脚本。此脚本可用于训练类条件 DiT 模型，但可以轻松修改以支持其他类型的条件。

在一个节点上使用 `1` 个 GPU 启动 DiT-XL/2 (256x256) 训练：

```bash
accelerate launch --mixed_precision fp16 train.py --model DiT-XL/2 --feature-path /path/to/store/features
```

在一个节点上使用 `N` 个 GPU 启动 DiT-XL/2 (256x256) 训练：
```bash
accelerate launch --multi_gpu --num_processes N --mixed_precision fp16 train.py --model DiT-XL/2 --feature-path /path/to/store/features
```

或者，您可以选择提取和训练 [训练选项](train_options) 文件夹中的脚本。


### PyTorch 训练结果

我们使用 PyTorch 训练脚本从头开始训练了 DiT-XL/2 和 DiT-B/4 模型，以验证它是否能够重现原始 JAX 结果，直到数十万次训练迭代。在我们的实验中，与 JAX 训练的模型相比，PyTorch 训练的模型给出了相似（有时略好）的结果，在合理的随机变化范围内。一些数据点：

| DiT 模型  | 训练步骤 | FID-50K<br> (JAX 训练) | FID-50K<br> (PyTorch 训练) | PyTorch 全局训练种子 |
|------------|-------------|----------------------------|--------------------------------|------------------------------|
| XL/2       | 400K        | 19.5                       | **18.1**                       | 42                           |
| B/4        | 400K        | **68.4**                   | 68.9                           | 42                           |
| B/4        | 400K        | 68.4                       | **68.3**                       | 100                          |

这些模型在 256x256 分辨率下训练；我们使用了 8 个 A100 来训练 XL/2，4 个 A100 来训练 B/4。请注意，此处的 FID 是使用 250 个 DDPM 采样步骤，使用 `mse` VAE 解码器且没有引导 (`cfg-scale=1`) 计算的。


### 改进的训练性能
与原始实现相比，我们实现了一系列训练速度加速和内存节省功能，包括梯度检查点、混合精度训练和预提取 VAE 特征，导致 DiT-XL/2 的速度提高了 95%，内存减少了 60%。使用 A100 全局批量大小为 128 的一些数据点：
 
| 梯度检查点 | 混合精度训练 | 特征预提取 | 训练速度 | 内存 |
|:----------------------:|:------------------------:|:----------------------:|:--------------:|:------------:|
| ❌                    | ❌                       | ❌                    | -              | 内存不足 |
| ✔                     | ❌                       | ❌                    | 0.43 steps/sec | 44045 MB     |
| ✔                     | ✔                        | ❌                    | 0.56 steps/sec | 40461 MB     |
| ✔                     | ✔                        | ✔                     | 0.84 steps/sec | 27485 MB     |


## 评估 (FID, Inception Score 等)

我们包含了一个 [`sample_ddp.py`](sample_ddp.py) 脚本，可以并行地从 DiT 模型中采样大量图像。这个脚本生成一个样本文件夹以及一个 `.npz` 文件，可以直接与 [ADM 的 TensorFlow 评估套件](https://github.com/openai/guided-diffusion/tree/main/evaluations) 一起使用，以计算 FID、Inception Score 和其他指标。例如，要在 `N` 个 GPU 上从我们预训练的 DiT-XL/2 模型中采样 50K 图像，请运行：

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model DiT-XL/2 --num-fid-samples 50000
```

还有几个额外的选项；详情请参见 [`sample_ddp.py`](sample_ddp.py)。


## 引用

```bibtex
@misc{jin2024fast,
    title={Fast-DiT: Fast Diffusion Models with Transformers},
    author={Jin, Chuanyang and Xie, Saining},
    howpublished = {\url{https://github.com/chuanyangjin/fast-DiT}},
    year={2024}
}
```
