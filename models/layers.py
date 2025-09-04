import torch
import torch.nn as nn
from typing import Optional


class _GradReverseFn(torch.autograd.Function):
    """Identity in the forward pass, sign-flips and scales the gradient on the way back."""
    @staticmethod
    def forward(ctx, x, λ: float):
        ctx.λ = λ
        return x

    @staticmethod
    def backward(ctx, g):
        return -ctx.λ * g, None     # no grad for λ


class GradReverse(nn.Module):
    """
    Gradient-reversal layer (DANN / adversarial learning).

    Parameters
    ----------
    λ : float
        Scale factor for the reversed gradient.
    """
    def __init__(self, λ: float = 0.1):
        super().__init__()
        self.λ = λ

    def forward(self, x):
        return _GradReverseFn.apply(x, self.λ)

    def set_lambda(self, λ: float):
        self.λ = λ


class HypersphereScaler(nn.Module):

    def __init__(self, radius=1, latent_dim=3, uniform_prior=True):
        super().__init__()
        self.radius = radius
        self.latent_dim = latent_dim
        self.uniform_prior = uniform_prior
    
    def forward(self, x):

        radius = x.norm(dim=-1, keepdim=True)

        # normalized_radius = torch.tanh(radius)
        normalized_radius = radius / (1 + radius)

        if self.uniform_prior: normalized_radius = normalized_radius.pow(1.0 / self.latent_dim)
        scale_factor = normalized_radius / (radius + 1e-9)
        
        return scale_factor * x * self.radius



class HypersphereLimiter(nn.Module):

    def __init__(self, radius=1):
        super().__init__()
        self.radius = radius
    
    def forward(self, x):
        radius = x.norm(dim=-1, keepdim=True)        
        return x / (radius + 1e-9) * self.radius

