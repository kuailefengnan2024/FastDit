# 文件功能：高斯扩散模型的核心实现
# 本文件提供了扩散模型的完整实现，包括前向扩散过程、反向采样过程、各种采样方法（如DDIM）、
# 损失计算以及其他扩散模型的核心功能。这是整个扩散模型框架的基础。
# 
# 修改自OpenAI的扩散模型代码库
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


import math

import numpy as np
import torch as th
import enum

from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl


def mean_flat(tensor):
    """
    对所有非批量维度求平均值。
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class ModelMeanType(enum.Enum):
    """
    模型预测的输出类型。
    """

    PREVIOUS_X = enum.auto()  # 模型预测x_{t-1}
    START_X = enum.auto()  # 模型预测x_0
    EPSILON = enum.auto()  # 模型预测噪声epsilon


class ModelVarType(enum.Enum):
    """
    用作模型输出方差的类型。
    添加了LEARNED_RANGE选项，允许模型预测介于FIXED_SMALL和FIXED_LARGE之间的值，使其任务更容易。
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # 使用原始MSE损失（学习方差时使用KL散度）
    RESCALED_MSE = (
        enum.auto()
    )  # 使用原始MSE损失（学习方差时使用RESCALED_KL）
    KL = enum.auto()  # 使用变分下界
    RESCALED_KL = enum.auto()  # 类似KL，但重新缩放以估计完整的VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    这是创建beta调度的已弃用API。
    请参阅get_named_beta_schedule()获取新的调度库。
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    获取给定名称的预定义beta调度。
    beta调度库由在num_diffusion_timesteps的极限下保持相似的beta调度组成。
    可以添加beta调度，但一旦提交，不应删除或更改，以保持向后兼容性。
    """
    if schedule_name == "linear":
        # Ho等人的线性调度，扩展为适用于任意数量的扩散步骤。
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"未知的beta调度: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    创建一个beta调度，将给定的alpha_t_bar函数离散化，
    该函数定义了从t = [0,1]的时间上(1-beta)的累积乘积。
    :param num_diffusion_timesteps: 要生成的beta数量。
    :param alpha_bar: 一个lambda函数，接受从0到1的参数t，
                      并生成直到扩散过程中该部分的(1-beta)的累积乘积。
    :param max_beta: 要使用的最大beta；使用小于1的值以防止奇点。
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    """
    训练和采样扩散模型的工具。
    原始代码来自此代码库：
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: 每个扩散时间步的beta的一维numpy数组，
                  从T开始到1结束。
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type
    ):

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        # 使用float64以提高精度。
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas必须是一维的"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # q(x_t | x_{t-1})和其他计算
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # 后验q(x_{t-1} | x_t, x_0)的计算
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # 以下：对数计算被截断，因为在扩散链开始时后验方差为0
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        获取分布q(x_t | x_0)。
        :param x_start: 形状为[N x C x ...]的无噪声输入张量。
        :param t: 扩散步数（减1）。这里，0表示一个步骤。
        :return: 三元组(mean, variance, log_variance)，所有元素都具有x_start的形状。
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        对数据进行给定扩散步数的扩散。
        换句话说，从q(x_t | x_0)采样。
        :param x_start: 初始数据批次。
        :param t: 扩散步数（减1）。这里，0表示一个步骤。
        :param noise: 如果指定，则为分离出的正态噪声。
        :return: x_start的噪声版本。
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        计算扩散后验的均值和方差：
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        将模型的输出应用于获取p(x_{t-1} | x_t)的均值和方差。
        :param model: 模型，接受x_t和t并返回ModelMeanType的相应输出。
        :param x: 形状为[N x C x ...]的输入数据
        :param t: 形状为[N]的包含时间索引的张量。
        :param clip_denoised: 如果为True，将去噪样本裁剪到[-1, 1]范围内。
        :param denoised_fn: 如果不为None，则是接受去噪的x并返回修改版本的函数。
        :param model_kwargs: 如果不为None，则为传递给模型的额外关键字参数字典。
        :return: 具有均值和方差的字典。
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # 控制范围的比例，确保model_var_values在[0, 1]区间，然后
                # 将其重新缩放到[min_var, max_var]区间。
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            if self.model_var_type == ModelVarType.FIXED_LARGE:
                model_variance = _extract_into_tensor(
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    t,
                    x.shape
                )
                model_log_variance = _extract_into_tensor(
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                    t,
                    x.shape
                )
            elif self.model_var_type == ModelVarType.FIXED_SMALL:
                model_variance = _extract_into_tensor(
                    self.posterior_variance, t, x.shape
                )
                model_log_variance = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
            else:
                raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            model_mean = model_output
        elif self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(pred_xstart)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        根据x_t和预测的噪声估计x_0。
        """
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        根据x_t和预测的x_0计算噪声残差。
        """
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        通过cond_fn对p_mean_var的均值进行外部条件化。
        """
        if cond_fn is None:
            return p_mean_var

        if model_kwargs is None:
            model_kwargs = {}

        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return {
            "mean": new_mean,
            "variance": p_mean_var["variance"],
            "log_variance": p_mean_var["log_variance"],
            "pred_xstart": p_mean_var["pred_xstart"],
        }

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        通过cond_fn对p_mean_var的EpsNet分数进行外部条件化。
        """
        if cond_fn is None:
            return p_mean_var

        if model_kwargs is None:
            model_kwargs = {}

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, t, **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        从p(x_{t-1} | x_t)采样。
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # t=0时不添加噪声
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        使用p_sample生成样本。
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        渐进式生成样本（每次返回所有采样的中间步骤）。
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is None:
            noise = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # 乐观地假设它们是整齐、均匀的步骤
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        img = noise
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        使用DDIM采样步骤从x_t生成样本。

        与p_sample相同，但使用DDIM更新规则。
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # 通常，我们可以从beta_t和model_variance得到
        # x_{t-1} = (x_t - coef1*eps) / coef2
        #
        # 这里，我们需要从x_0的预测值添加更多噪声，这样
        # x_{t-1} = (x_t - coef1*eps) / coef2 + noise_amount*z
        #
        # 其中z是标准高斯样本
        # 在DDIM中，噪声根据eta参数调整

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # t=0时不添加噪声
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        使用DDIM逆采样步骤从x_{t-1}获得x_t样本。
        """
        assert eta == 0.0, "反向采样仅适用于确定性DDIM"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # 通常，我们可以从beta_t和model_variance得到
        # x_{t-1} = (x_t - coef1*eps) / coef2
        #
        # 这里，我们需要从x_0的预测值中获取x_t，并非x_{t-1}。
        # x_t = coef3*x_0 + coef4*eps
        #
        # 其中eps通过x_{t-1}和x_0计算得出

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        sample = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        使用DDIM生成样本。
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        使用DDIM渐进式生成样本。
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is None:
            noise = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # 乐观地假设它们是整齐、均匀的步骤
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        img = noise
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        计算VLB中的KL项，以位/维度为单位。
        
        这里包括的条款是：
            KL(q(x_{t-1} | x_t, x_0) || p(x_{t-1} | x_t))
        
        x_t对于前向过程是已知的，因此没有任何熵损失。
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # 在第一步（t=0）中，使用解码器NLL而不是KL散度
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        计算训练损失。
        
        :param model: 要评估的模型。
        :param x_start: [N x C x ...] 张量，包含样本批次 x_0。
        :param t: 包含扩散步长索引的一维张量。
        :param model_kwargs: 如果不为None，则为要传递给模型的额外关键字参数字典。
        :param noise: 如果指定，则为用于添加到样本x_0的噪声张量。
        :return: 包含各种损失值的字典。
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, t, **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # 学习方差值是无约束的实数，因此在计算损失之前将其传递给噪声预测模型可能会产生问题
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # 乘以代表重新加权VLB项的数量，如公式（16）所示
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        计算完整VLB中的先验KL项。

        这项是：
            KL(q(x_{T-1} | x_0) || p(x_{T-1}))
        
        其中 p(x_{T-1}) 是标准高斯分布，q(x_{T-1} | x_0) 是 x_0 的高斯扩散后的结果。
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        计算整个扩散过程的比特/维度的变分下界。
        
        :param model: 要评估的模型。
        :param x_start: 包含样本批次x_0的数据。
        :param clip_denoised: 如果为True，则将去噪的x_0预测裁剪到[-1, 1]范围内。
        :param model_kwargs: 如果不为None，则为要传递给模型的额外关键字参数字典。
        :return: 包含 bpd 和每个 ts 的 K-L 发散的字典。
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # 计算VLB项
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    从一个一维数组提取值，并将它们重新整形为具有给定形状的张量
    添加一个批次维度，然后广播到给定的形状。
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
