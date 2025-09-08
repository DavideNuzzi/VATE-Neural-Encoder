import math
import torch
import torch.nn as nn

LOG2PI = math.log(2.0 * math.pi) 


class AR1Transition(nn.Module):

    def __init__(self,
                 latent_dim: int,
                 *,
                 init_rho: float = 0.9,
                 learn_stationary_var: bool = True,
                 init_stationary_var: float = 1.0,
                 logvar_lims: tuple[float, float] = (-9.0, 2.0)):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.logvar_lims = logvar_lims

        init_rho = float(max(min(init_rho, 0.99), -0.99))
        raw_rho0 = 0.5 * math.log((1.0 + init_rho) / (1.0 - init_rho))
        self.raw_rho = nn.Parameter(torch.full((self.latent_dim,), raw_rho0))

        if learn_stationary_var:
            self.log_svar = nn.Parameter(torch.full((self.latent_dim,), math.log(init_stationary_var)))
        else:
            self.register_buffer("log_svar", torch.full((self.latent_dim,), math.log(init_stationary_var)), persistent=False)

    def forward(self, z: torch.Tensor):
        # z: (..., T, L)
        shape = (1,) * (z.dim() - 1) + (self.latent_dim,)
        rho  = torch.tanh(self.raw_rho).view(shape)          # (..., L)
        svar = torch.exp(self.log_svar).view(shape)          # (..., L)

        mu = rho * z                                         # (..., T, L)
        var = torch.clamp(svar * (1.0 - rho.pow(2)), min=1e-12)
        logvar = torch.log(var).clamp(*self.logvar_lims).expand_as(mu)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z_next = mu + std * eps

        return z_next, mu, logvar

    def log_prob(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * (((z - mu).pow(2) * torch.exp(-logvar)) + logvar + LOG2PI).sum(dim=-1)



class LeakyResidualTransition(nn.Module):

    def __init__(self,
                 *,
                 body_module: nn.Module,     # maps (..., T, L) -> (..., T, H)
                 latent_dim: int,
                 logvar_lims: tuple[float, float] = (-9.0, 2.0),
                 init_alpha: float = 0.9,     # carry-over coefficient in (0,1)
                 init_res_scale: float = 0.05):
        super().__init__()
        self.body_module = body_module
        self.latent_dim = int(latent_dim)
        self.logvar_lims = logvar_lims

        self.delta_layer  = nn.LazyLinear(self.latent_dim)   # bounded residual head
        self.logvar_layer = nn.LazyLinear(self.latent_dim)   # innovation variance head

        init_alpha = float(min(max(init_alpha, 0.01), 0.99))
        self.raw_alpha = nn.Parameter(torch.full((self.latent_dim,),
                                   math.log(init_alpha) - math.log(1.0 - init_alpha)))
        self.res_scale = nn.Parameter(torch.full((self.latent_dim,), init_res_scale))

    def forward(self, z: torch.Tensor):
        # z: (..., T, L)
        h = self.body_module(z)                               # (..., T, H)
        delta = self.delta_layer(h)                           # (..., T, L)

        shape = (1,) * (z.dim() - 1) + (self.latent_dim,)
        alpha = torch.sigmoid(self.raw_alpha).view(shape)     # (..., L) in (0,1)
        mu = alpha * z + torch.tanh(delta) * self.res_scale.view(shape)

        logvar = self.logvar_layer(h).clamp(*self.logvar_lims)
        logvar = logvar.expand_as(mu)
        std = torch.exp(0.5 * logvar)
        z_next = mu + std * torch.randn_like(mu)
        return z_next, mu, logvar

    def log_prob(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * (((z - mu).pow(2) * torch.exp(-logvar)) + logvar + LOG2PI).sum(dim=-1)