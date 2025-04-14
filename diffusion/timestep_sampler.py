# 文件功能：扩散模型时间步采样器的实现
# 本文件实现了多种时间步采样策略，用于训练扩散模型时的时间步选择。
# 包括均匀采样和基于损失加权的采样方法，以减少训练过程中的方差。
#
# 修改自OpenAI的扩散模型代码库
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    从预定义的采样器库中创建一个ScheduleSampler。
    :param name: 采样器的名称。
    :param diffusion: 要采样的扩散对象。
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"未知的调度采样器: {name}")


class ScheduleSampler(ABC):
    """
    扩散过程中时间步的分布，旨在减少目标函数的方差。
    
    默认情况下，采样器执行无偏重要性采样，其中目标的均值保持不变。
    然而，子类可以覆盖sample()方法来改变重新采样项的重新加权方式，
    从而允许目标函数实际发生变化。
    """

    @abstractmethod
    def weights(self):
        """
        获取每个扩散步骤的权重的numpy数组。
        权重不需要归一化，但必须为正值。
        """

    def sample(self, batch_size, device):
        """
        对一批数据进行时间步的重要性采样。
        :param batch_size: 时间步的数量。
        :param device: 保存的torch设备。
        :return: 一个元组(timesteps, weights):
                 - timesteps: 时间步索引的张量。
                 - weights: 用于缩放结果损失的权重张量。
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        使用模型的损失更新重新加权。
        
        从每个进程调用此方法，传入一批时间步和对应于每个时间步的损失。
        此方法将执行同步，以确保所有进程保持完全相同的重新加权。
        
        :param local_ts: 时间步的整数张量。
        :param local_losses: 损失的1D张量。
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # 将all_gather批次填充到最大批次大小。
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        使用模型的损失更新重新加权。
        
        子类应覆盖此方法，使用模型的损失更新重新加权。
        
        此方法直接更新重新加权，而不在工作进程之间同步。
        它由所有等级的update_with_local_losses调用，参数相同。
        因此，它应具有确定性行为，以维持跨工作者的状态。
        
        :param ts: 整数时间步列表。
        :param losses: 浮点损失列表，每个时间步一个。
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # 移出最旧的损失项。
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
