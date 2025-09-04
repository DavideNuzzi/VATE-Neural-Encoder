import math
import torch
import math
from typing import Dict, Tuple, Optional, Callable


TAU = 2.0 * math.pi

def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    return torch.atan2(torch.sin(a), torch.cos(a))

def angles_to_torus(theta: torch.Tensor, phi: torch.Tensor, R: float = 2.0, r: float = 0.7) -> torch.Tensor:
    """theta, phi: (T,) -> z: (T, 3) embedded on a torus with major R and minor r."""
    ct, st = torch.cos(theta), torch.sin(theta)
    cp, sp = torch.cos(phi), torch.sin(phi)
    x = (R + r * cp) * ct
    y = (R + r * cp) * st
    z = r * sp
    return torch.stack([x, y, z], dim=-1)  # (T,3)

def simulate_torus_latents(
    T: int,
    *,
    dt: float = 0.1,
    omega: Tuple[float, float] = (0.35, 0.12),
    sigma: Tuple[float, float] = (0.03, 0.03),
    R: float = 2.0,
    r: float = 0.7,
    subsample: int = 1,
    seed: Optional[int] = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Simulate a single trajectory on a torus.
    Returns theta:(T,), phi:(T,), z:(T,3) on `device`.
    """
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    # initial angles in [-pi, pi)
    theta0 = TAU * torch.rand((), device=device, generator=g, dtype=dtype) - math.pi
    phi0   = TAU * torch.rand((), device=device, generator=g, dtype=dtype) - math.pi

    if T < 1:
        raise ValueError("T must be >= 1")

    if T == 1:
        theta = torch.tensor([theta0], device=device, dtype=dtype)
        phi   = torch.tensor([phi0],   device=device, dtype=dtype)
    else:
        # increments
        eps_th = torch.randn(T-1, device=device, generator=g, dtype=dtype)
        eps_ph = torch.randn(T-1, device=device, generator=g, dtype=dtype)
        dtheta = omega[0] * dt + sigma[0] * math.sqrt(dt) * eps_th
        dphi   = omega[1] * dt + sigma[1] * math.sqrt(dt) * eps_ph

        # cumulative angles, then wrap
        theta = torch.empty(T, device=device, dtype=dtype)
        phi   = torch.empty(T, device=device, dtype=dtype)
        theta[0] = theta0
        phi[0]   = phi0
        theta[1:] = theta0 + torch.cumsum(dtheta, dim=0)
        phi[1:]   = phi0   + torch.cumsum(dphi,   dim=0)
        theta = wrap_angle(theta)
        phi   = wrap_angle(phi)

    theta = theta[::subsample]
    phi = phi[::subsample]
    
    z = angles_to_torus(theta, phi, R=R, r=r)  # (T,3)
    return {"theta": theta, "phi": phi, "z": z}

def sector_labels_from_theta(theta: torch.Tensor, C: int = 8) -> torch.Tensor:
    """Discretize theta (T,) into C sectors -> (T,) long in [0..C-1]."""
    theta_mod = (theta % TAU + TAU) % TAU              # [0, 2pi)
    bins = torch.floor(theta_mod / (TAU / C)).long()
    return torch.clamp(bins, 0, C-1)

def random_feature_map(
    z: torch.Tensor,                      # (T,3)
    F: int,
    *,
    hidden: int = 128,
    depth: int = 2,
    nonlin: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    noise_std: float = 0.05,
    seed: Optional[int] = 1,
) -> torch.Tensor:
    """
    Fixed random MLP: z(T,3) -> x(T,F). Weights are sampled (reproducibly) and NOT learned.
    """
    device, dtype = z.device, z.dtype
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    h = z  # (T,3)
    in_dim = h.shape[-1]
    for _ in range(depth):
        W = torch.randn(in_dim, hidden, device=device, generator=g, dtype=dtype) / math.sqrt(in_dim)
        b = torch.randn(1, hidden, device=device, generator=g, dtype=dtype) * 0.1
        h = nonlin(h @ W + b)
        in_dim = hidden

    W_out = torch.randn(in_dim, F, device=device, generator=g, dtype=dtype) / math.sqrt(in_dim)
    b_out = torch.randn(1, F, device=device, generator=g, dtype=dtype) * 0.1
    x = h @ W_out + b_out  # (T,F)

    if noise_std > 0:
        noise = torch.randn(x.shape, device=device, dtype=dtype, generator=g)
        x = x + noise_std * noise
    return x

def make_torus_timeseries(
    T: int,
    F: int,
    *,
    dt: float = 0.1,
    omega: Tuple[float, float] = (0.35, 0.12),
    sigma: Tuple[float, float] = (0.03, 0.03),
    R: float = 2.0,
    r: float = 0.7,
    C: int = 8,
    subsample: int = 1,
    feature_hidden: int = 128,
    feature_depth: int = 2,
    feature_noise: float = 0.05,
    seed_latent: Optional[int] = 0,
    seed_feature: Optional[int] = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Produce a single full timeseries:
      neural:  (T,F)
      targets: {'behavior': (T,)}
      latents: (T,3)
      angles:  {'theta': (T,), 'phi': (T,)}
    """
    sim = simulate_torus_latents(
        T, dt=dt, omega=omega, sigma=sigma, R=R, r=r,
        subsample=subsample, seed=seed_latent, device=device, dtype=dtype,
    )

    labels = sector_labels_from_theta(sim["theta"], C=C)        # (T,)
    x = random_feature_map(
        sim["z"], F, hidden=feature_hidden, depth=feature_depth,
        noise_std=feature_noise, seed=seed_feature,
    )   
                                                            # (T,F)
    return {
        "neural": x,
        "targets": {"behavior": labels.to(torch.long)},
        "latents": sim["z"],
        "angles": {"theta": sim["theta"], "phi": sim["phi"]},
    }