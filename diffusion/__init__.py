# 文件功能：扩散模型包的初始化文件
# 本文件提供了创建扩散模型实例的便捷函数，并导入各个模块的主要组件。
# 这是整个扩散模型包的入口点，提供了一个易于使用的API来创建和配置扩散模型。
#
# 修改自OpenAI的扩散模型代码库
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    """
    创建一个扩散模型实例。
    
    :param timestep_respacing: 时间步重新采样的配置，可以是空字符串（使用所有步骤）或特定的采样模式
    :param noise_schedule: 噪声调度类型，如"linear"或"squaredcos_cap_v2"
    :param use_kl: 是否使用KL散度作为损失函数
    :param sigma_small: 是否使用较小的固定方差
    :param predict_xstart: 模型是否直接预测x_0（而不是预测噪声）
    :param learn_sigma: 是否学习方差参数
    :param rescale_learned_sigmas: 是否重新缩放学习到的方差
    :param diffusion_steps: 扩散过程中的步骤总数
    :return: 配置好的SpacedDiffusion实例
    """
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )
