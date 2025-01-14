import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import UNet
from functools import partial
from copy import deepcopy


# 提取对应的timestep的时间编码
def extract(a, t, x_shape):
    # a:       schedule   []
    # t:       timestep   [index * batch]
    # x_shape: imageshape [batch, c, h, w]

    b, *_ = t.shape
    out = a.gather(-1, t) # 在a的最后维度上取index为t的值 [B]
    print(((1,) * (len(x_shape) - 1)))
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # [B, 1]

batch = 10
t = extract(torch.arange(1000), torch.tensor([999] * batch), torch.ones(batch, 3, 64, 64))

#---------------------------------------------------------#
#   ema_model = decay * ema_model + (1 - decay) * model
#---------------------------------------------------------#
class EMA():
    def __init__(self, decay):
        self.decay = decay

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)

class GaussianDiffusion(nn.Module):
    def __init__(
        self, model: UNet, img_size, img_channels, num_classes=None, betas=[], loss_type="l2", ema_decay=0.9999, ema_start=2000, ema_update_rate=1,
    ):
        super().__init__()
        self.model      = model
        self.ema_model  = deepcopy(model)

        self.ema                = EMA(ema_decay)
        self.ema_decay          = ema_decay
        self.ema_start          = ema_start
        self.ema_update_rate    = ema_update_rate
        self.step               = 0

        self.img_size       = img_size
        self.img_channels   = img_channels
        self.num_classes    = num_classes

        # l1或者l2损失
        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type      = loss_type
        self.num_timesteps  = len(betas)

        alphas              = 1.0 - betas
        alphas_cumprod      = np.cumprod(alphas) # 累乘,返回序列

        # 转换成torch.tensor来处理
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # betas             [0.0001, 0.00011992, 0.00013984 ... , 0.02]
        self.register_buffer("betas", to_torch(betas))
        # alphas            [0.9999, 0.99988008, 0.99986016 ... , 0.98]
        self.register_buffer("alphas", to_torch(alphas))
        # alphas_cumprod    [9.99900000e-01, 9.99780092e-01, 9.99640283e-01 ... , 4.03582977e-05]
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        # x_{t} = \sqrt{\alpha_{t}} x_{t - 1} + \sqrt{1 - \alpha_{t}} z_{1}
        # sqrt(alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        # sqrt(1 - alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))

        # x_{t - 1}
        # =
        # \frac{1}{\sqrt{\alpha_t}}
        # \left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta\left(x_t, t\right)\right)
        # +
        # \sigma_t z
        # sqrt(1 / alphas)
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        # \sigma_t z
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    @torch.no_grad()
    def remove_noise(self, x, t, y, use_ema=True):
        """移除噪声

        .. math::
            \frac{1}{\sqrt{\alpha_t}}
            \times
            \left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta\left(x_t, t\right)\right)
        """
        if use_ema:
            return (    # (x - remove_noise_coeff * pred_noise) * reciprocal_sqrt_alphas
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )

    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True):
        """随机生成图片,按照时间步逐步移除噪声

        .. math::
            x_{t - 1}
            =
            \frac{1}{\sqrt{\alpha_t}}
            \times
            \left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta\left(x_t, t\right)\right)
            +
            \sigma_t z
        """
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        # 随机生成输入
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size) # 999 998 ... [999] => [999 * batch_size]
            # 移除噪音
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                # + \sigma_t z
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

        return x.cpu().detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        """随机生成图片,按照时间步逐步移除噪声,返回生成的序列"""
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        # 随机生成输入
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size) # 999 998 ... [999] => [999 * batch_size]
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                # + \sigma_t z
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.cpu().detach())

        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        """加噪音

        .. math::
            x_{t} = \sqrt{\alpha_{t}} x_{t - 1} + \sqrt{1 - \alpha_{t}} z_{1}
        """
        return (
            extract(self.sqrt_alphas_cumprod, t,  x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def get_losses(self, x, t, y):
        # x, noise: [batch_size, 3, 64, 64]
        # t:        [batch_size]
        # y:        None
        noise           = torch.randn_like(x)
        torch.nn.MSELoss
        perturbed_x     = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, t, y)
        torch.nn.MSELoss
        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)
        return loss

    def forward(self, x, y=None):
        b, c, h, w  = x.shape
        device      = x.device

        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[0]:
            raise ValueError("image width does not match diffusion parameters")

        # 使用生成随机的timestep训练
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.get_losses(x, t, y)

def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)

    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    return np.array(betas)

def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    unet = UNet(img_channels = 3, base_channels = 128)

    net = GaussianDiffusion(
        model=unet,
        img_size=[64, 64],
        img_channels=3,
        betas=generate_linear_schedule(T = 1000, low= 1e-4 * 1000 / 1000, high = 0.02 * 1000 / 1000)
    ).to(device)

    x = torch.ones(1, 3, 64, 64).to(device)
    loss = net(x)
    print(loss) # 1.1342

    images = net.sample(batch_size=1, device=device)
    print(images.shape) # [batch_size, 3, 64, 64]
