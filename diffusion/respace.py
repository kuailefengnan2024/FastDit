# 文件功能：提供扩散模型时间步重采样功能
# 本文件实现了扩散模型时间步的重新采样和间隔调整，允许从原始扩散过程中
# 选择特定的时间步进行训练或推理，比如DDIM采样。这可以加速采样过程或改变
# 扩散模型的行为。
#
# 修改自OpenAI的扩散模型代码库
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    从原始扩散过程中创建一个时间步列表，
    给定我们想要从原始过程的等大小部分中采取的时间步数量。
    
    例如，如果有300个时间步，而section_counts是[10,15,20]，
    那么前100个时间步会被调整为10个时间步，中间100个
    会被调整为15个时间步，最后100个会被调整为20个时间步。
    
    如果步幅是以"ddim"开头的字符串，则使用DDIM论文中的固定步幅，
    且只允许一个部分。
    
    :param num_timesteps: 原始过程中要分割的扩散步骤数。
    :param section_counts: 可以是数字列表，或包含逗号分隔数字的字符串，
                         表示每个部分的步骤数。作为一种特殊情况，使用"ddimN"
                         其中N是步数，使用DDIM论文中的步幅。
    :return: 要使用的原始过程中的一组扩散步骤。
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"无法使用整数步幅创建恰好{num_timesteps}个步骤"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"无法将{size}个步骤的部分分成{section_count}个"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    一种可以在基本扩散过程中跳过步骤的扩散过程。
    
    :param use_timesteps: 从原始扩散过程中保留的时间步集合（序列或集合）。
    :param kwargs: 创建基本扩散过程的关键字参数。
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # 缩放由包装后的模型完成。
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        # self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        # if self.rescale_timesteps:
        #     new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
