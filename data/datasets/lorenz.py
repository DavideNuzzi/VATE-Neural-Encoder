# lorenz_dataset.py
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Union, Tuple


def _lorenz_rhs(t: float, x: np.ndarray,
                rho: float, sigma: float, beta: float) -> np.ndarray:
    """dx/dt for the classic Lorenz system."""
    return np.array([
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ])


def simulate_lorenz(x0: np.ndarray,
                    t: np.ndarray,
                    rho: float = 28.0,
                    sigma: float = 10.0,
                    beta: float = 8.0/3.0) -> np.ndarray:

    sol = solve_ivp(
        fun=lambda _t, _x: _lorenz_rhs(_t, _x, rho, sigma, beta),
        t_span=(float(t[0]), float(t[-1])),
        y0=x0,
        t_eval=t,
        method="RK45",
    )
    return sol.y.T         


def get_lorenz_labels(x):
    
    labels = np.zeros(x.shape[0], dtype=np.int64)
    labels[x[:,0] > 0] = 1
    return labels


def create_neural_timeseries(x_lorenz, channels=64, A=None):

    # x_lorenz = x_lorenz ** 2
    x_lorenz = (x_lorenz - np.mean(x_lorenz, axis=0, keepdims=True)) /  np.std(x_lorenz, axis=0, keepdims=True)
    # Create a basis of values obtained from the lorenz variables x,y,z
    x, y, z = x_lorenz[:,0], x_lorenz[:,1], x_lorenz[:,2]
    
    x_basis = np.column_stack([np.sin(x - y), np.sin(x - z), np.sin(y - z),
                               np.sin(x), np.sin(y), np.sin(z)
    ])

    # x_basis = x_lorenz

    if A is None:
        A = np.linalg.qr(np.random.randn(channels, x_basis.shape[1])).Q
    
    x_neural = A @ x_basis.T

    # x_neural = x_neural ** 3


    # x_neural = (x_neural - np.mean(x_neural, axis=0, keepdims=True)) /  np.std(x_neural, axis=0, keepdims=True)
    x_neural = (x_neural - np.mean(x_neural, axis=1, keepdims=True)) /  np.std(x_neural, axis=1, keepdims=True)

    # α = 0.8 + 0.4*np.random.rand(channels, 1)  # channel gains
    # β = 0.8 + 0.4*np.random.rand(channels, 1)  # slopes
    # γ = 0.2*np.random.randn(channels, 1)       # biases
    # x_neural = α * np.tanh(β * x_neural + γ)       # monotone warp

    return x_neural.T, A


# def rotate_trajectory(x, roll=0, pitch=0, yaw=0):

#     roll /= 180 * np.pi
#     pitch /= 180 * np.pi
#     yaw /= 180 * np.pi
   
#     ca, cb, cg = np.cos([roll, pitch, yaw])
#     sa, sb, sg = np.sin([roll, pitch, yaw])
    
#     R = np.array([
#         [cb * cg,              cb * sg,           -sb],
#         [sa * sb * cg - ca * sg, sa * sb * sg + ca * cg, sa * cb],
#         [ca * sb * cg + sa * sg, ca * sb * sg - sa * cg, ca * cb]
#     ])

#     return x @ R.T
