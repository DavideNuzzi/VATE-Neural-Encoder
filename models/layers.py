import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from abc import ABC, abstractmethod
import torch.nn.functional as F

LOG2PI = math.log(2.0 * math.pi) 

def standardize_shape(x: torch.Tensor) -> torch.Tensor:
    """Convert 1D/2D input into (batch, time, features)."""
    if x.ndim == 1:   # (F,)
        return x.unsqueeze(0).unsqueeze(0)   # (1,1,F)
    if x.ndim == 2:   # (T,F)
        return x.unsqueeze(0)                # (1,T,F)
    return x

def pack_leading_dims(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    """Flatten all leading dims except the last two into batch."""
    shape = x.shape
    x = x.view(-1, *shape[-2:])   # (B*, T, F)
    return x, shape

def unpack_leading_dims(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Restore original leading dims after merging."""
    return x.view(*shape[:-2], *x.shape[-2:])


# ------------------------------ Gaussian Layer ------------------------------ #
class GaussianLayer(nn.Module):
    def __init__(self,
                 output_dim: int,
                 *,
                 logvar_lims: Tuple[float, float] = (-12.0, -3.0),
                 fixed_logvar: Optional[float] = None):
        
        super().__init__()
        self.output_dim = output_dim
        self.fixed_logvar = fixed_logvar
        self.logvar_lims = logvar_lims

        # Create the mean layer
        self.mean_layer = nn.LazyLinear(self.output_dim)

        # Create the logvar layer or fixed logvar buffer
        if self.fixed_logvar is None:
            self.logvar_layer = nn.LazyLinear(self.output_dim)
            self.register_buffer("logvar_const_buf", None, persistent=False)
        else:
            self.logvar_layer = None
            # self.logvar_const_buf: torch.Tensor #
            self.register_buffer(
                "logvar_const_buf",
                torch.full((1, self.output_dim), float(self.fixed_logvar))
            )
            
    def get_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Compute mean an logvar
        mu = self.mean_layer(x)                   

        if self.logvar_layer is not None:
            logvar = self.logvar_layer(x) 
        else:
            logvar = self.logvar_const_buf.expand_as(mu) # type: ignore

        logvar = torch.clamp(logvar, *self.logvar_lims)
        return mu, logvar
    
    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        
        # Sample with the reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps     
        return z
  
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Get mean, logvar and then get one sample
        mu, logvar = self.get_params(x)
        z = self.sample(mu, logvar)

        return z, mu, logvar           

    def log_prob(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

        return -0.5 * ((x - mu).pow(2) * torch.exp(-logvar) + logvar + LOG2PI).sum(dim=-1)



# ---------------------------------------------------------------------------- #
#                                  Transition                                  #
# ---------------------------------------------------------------------------- #

class Transition(nn.Module):
    
    def __init__(self,
                 *,
                 body_module: nn.Module, 
                 latent_dim: Optional[int] = None,
                 gaussian_module: Optional[GaussianLayer] = None,
                 residual_mode: str = 'none',
                 ):
        
        super().__init__()

        self.body_module = body_module
        self.residual_mode = residual_mode

        # Either latent_dim or gaussian_module must be provided
        if (latent_dim is None) == (gaussian_module is None):
            raise ValueError("Either latent_dim or gaussian_module must be provided, but not both.")
        if gaussian_module is not None:
            self.gaussian_module = gaussian_module
            self.latent_dim = gaussian_module.output_dim
        elif latent_dim is not None:
            self.gaussian_module = GaussianLayer(latent_dim)
            self.latent_dim = latent_dim
        
        # In residual_mode="bounded" the transition needs an additional
        # parameter for global step scale
        if residual_mode == "bounded":
            self.residual_scale = nn.Parameter(torch.tensor(0.1))


    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Apply the body module
        h = self.body_module(z)

        # If in default mode the parameters of the gaussian are computed
        # directly from h. If in "simple" mode, the mean is given by
        # z_{t-1} + h. If in "bounded" mode they are given by
        # z_{t-1} + tanh(h) * scale
        if self.residual_mode == 'none':
            z, mu, logvar = self.gaussian_module(h)
        else:
            mu, logvar = self.gaussian_module.get_params(h)

            if self.residual_mode == 'simple':
                mu = z + mu
            if self.residual_mode == 'bounded':
                mu = z + torch.tanh(mu) * self.residual_scale

            z = self.gaussian_module.sample(mu, logvar)

        return z, mu, logvar

    def log_prob(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

        return self.gaussian_module.log_prob(z, mu, logvar)
            
# ---------------------------------------------------------------------------- #
#                                   Encoders                                   #
# ---------------------------------------------------------------------------- #

# ------------------------------- Base Encoder ------------------------------- #
# This is the base abstract class for all encoders that deals with the Gaussian
# latents sampling and log-prob computation. It operates assuming that the input
# data has already been processed by some body module. 

class BaseEncoder(nn.Module, ABC):

    def __init__(self,
                 latent_dim: Optional[int] = None,
                 gaussian_module: Optional[GaussianLayer] = None):
        super().__init__()

        # Either latent_dim or gaussian_module must be provided
        if (latent_dim is None) == (gaussian_module is None):
            raise ValueError("Either latent_dim or gaussian_module must be provided, but not both.")
        if gaussian_module is not None:
            self.gaussian_module = gaussian_module
            self.latent_dim = gaussian_module.output_dim
        elif latent_dim is not None:
            self.gaussian_module = GaussianLayer(latent_dim)
            self.latent_dim = latent_dim

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def log_prob(self, z: torch.Tensor, *params) -> torch.Tensor:
        pass

# ---------------------------- Factorized Encoder ---------------------------- #
# This is just q(z_t | x_t) without any temporal context

class FactorizedEncoder(BaseEncoder):

    def __init__(self, *,
                 body_module: nn.Module,
                 latent_dim: Optional[int] = None,
                 gaussian_module: Optional[GaussianLayer] = None):

        super().__init__(latent_dim=latent_dim, gaussian_module=gaussian_module)
        self.body_module = body_module

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x = standardize_shape(x)

        # Encode each time step indipendently
        h = self.body_module(x) # (..., window_len, features) -> (..., window_len, hiddne_dim)
        
        # Sample from the Gaussian layer
        z, mu, logvar = self.gaussian_module(h) # (..., window_len, latent_dim)

        return z, mu, logvar
    
    # Computes log q(z | x), but maybe it is trivial and I should put it in the forward
    # or make the "x" an argument (so that I can do the full computation instead of passing z)
    def log_prob(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

        return self.gaussian_module.log_prob(z, mu, logvar)


# ----------------------- Recurrent Factorized Encoder ----------------------- #
# This is q(z_t | x_{0:t}) with a recurrent context. If the recurrent module
# is bidirectional, this stops being causal and becomes a smoother.

class RecurrentFactorizedEncoder(BaseEncoder):
    
    def __init__(self, *,
                 body_module: nn.Module,
                 recurrent_module: nn.Module,
                 latent_dim: Optional[int] = None,
                 gaussian_module: Optional[GaussianLayer] = None):

        super().__init__(latent_dim=latent_dim, gaussian_module=gaussian_module)
        self.body_module = body_module
        self.recurrent_module = recurrent_module

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Convert 1D or 2D tensors to the form (batch, time, features)
        x = standardize_shape(x)
        x, input_shape = pack_leading_dims(x)

        # Encode the whole sequence with a recurrent module
        h_rnn, _ = self.recurrent_module(x)

        # Unpack the leading dimensions
        h_rnn = unpack_leading_dims(h_rnn, input_shape)

        # Apply the body module to the RNN output
        h_body = self.body_module(h_rnn)
        
        # Sample from the Gaussian layer
        z, mu, logvar = self.gaussian_module(h_body) 
                                             
        return z, mu, logvar
    
    # Computes log q(z | x), but maybe it is trivial and I should put it in the forward
    # or make the "x" an argument (so that I can do the full computation instead of passing z)
    def log_prob(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

        return self.gaussian_module.log_prob(z, mu, logvar)



# -------------------------- Autoregressive Encoder -------------------------- #
# This is q(z_t | z_{t-1}, x_{0:t}) with a recurrent context and the gaussian
# mean and logvar are given by a function of the RNN output and the previous z directly
# or through a residual step.

class AutoregressiveEncoder(BaseEncoder):
    
    def __init__(self, *,
                 body_module: nn.Module,
                 recurrent_module: nn.Module,
                 latent_dim: Optional[int] = None,
                 gaussian_module: Optional[GaussianLayer] = None,
                 residual_mode: str = "none"):

        super().__init__(latent_dim=latent_dim, gaussian_module=gaussian_module)
        self.body_module = body_module
        self.recurrent_module = recurrent_module
        self.residual_mode = residual_mode

        # In residual_mode="bounded" the encoder needs an additional parameter for 
        # global step scale
        if residual_mode == "bounded":
            self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Convert 1D or 2D tensors to the form (batch, time, features)
        x = standardize_shape(x)
        x, input_shape = pack_leading_dims(x)
        B, T = x.shape[:2]

        # Encode the whole sequence with a recurrent module
        h_rnn, _ = self.recurrent_module(x)
        
        # Initalize the first latent and the list of latents
        # Note that for t = -1 we assume that z = 0
        z_seq, mu_seq, logvar_seq = [], [], []
        z_prev = torch.zeros((B, self.latent_dim), device=x.device, dtype=x.dtype)

        # Start the autoregressive posterior calculation
        for t in range(T):

            # Get the context at time t
            h_t = h_rnn[:, t, :]

            # Concatenate it to the previous sample (along the last dimension)
            s_t = torch.cat([h_t, z_prev], dim=-1)

            # Apply the body module
            s_t = self.body_module(s_t)

            # If in default mode the parameters of the gaussian are computed
            # directly from s_t. If in "simple" mode, the mean is given by
            # z_{t-1} + s_t.  If in "bounded" mode they are given by
            # z_{t-1} + tanh(h) * scale
            if self.residual_mode == 'none':
                z, mu, logvar = self.gaussian_module(s_t)
            else:
                mu, logvar = self.gaussian_module.get_params(s_t)

                if self.residual_mode == 'simple':
                    mu = z_prev + mu
                if self.residual_mode == 'bounded':
                    mu = z_prev + torch.tanh(mu) * self.residual_scale

                z = self.gaussian_module.sample(mu, logvar)

            # Save and go to the next iteration
            z_seq.append(z)
            mu_seq.append(mu)
            logvar_seq.append(logvar)

            # log_q_seq.append(log_q)
            z_prev = z  

        # Stack the lists to get the time dimension
        z_seq = torch.stack(z_seq, dim=1)
        mu_seq = torch.stack(mu_seq, dim=1)
        logvar_seq = torch.stack(logvar_seq, dim=1)

        # Unpack the leading dimensions
        z_seq = unpack_leading_dims(z_seq, input_shape)
        mu_seq = unpack_leading_dims(mu_seq, input_shape)
        logvar_seq = unpack_leading_dims(logvar_seq, input_shape)

        return z_seq, mu_seq, logvar_seq
    
    # Computes log q(z | x), but maybe it is trivial and I should put it in the forward
    # or make the "x" an argument (so that I can do the full computation instead of passing z)
    def log_prob(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

        return self.gaussian_module.log_prob(z, mu, logvar)



# ------------------------- Prior Controlled Encoder ------------------------- #
# This is q(z_t | z_{t-1}, x_{0:t}) with a recurrent context and the gaussian
# mean and logvar are given by a function of the RNN output and the previous z directly.

class PriorControlledEncoder(BaseEncoder):
    
    def __init__(self, *,
                 body_module: nn.Module,
                 recurrent_module: nn.Module,
                 transition_module: Transition,
                 latent_dim: Optional[int] = None,
                 gaussian_module: Optional[GaussianLayer] = None):

        super().__init__(latent_dim=latent_dim, gaussian_module=gaussian_module)
        self.body_module = body_module
        self.recurrent_module = recurrent_module
        self.transition_module = transition_module

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Convert 1D or 2D tensors to the form (batch, time, features)
        x = standardize_shape(x)
        x, input_shape = pack_leading_dims(x)
        B, T = x.shape[:2]

        # Encode the whole sequence with a recurrent module
        h_rnn, _ = self.recurrent_module(x)
        
        # Initalize the first latent and the list of latents
        # Note that for t = -1 we assume that z = 0
        z_seq, mu_seq, logvar_seq = [], [], []
        z_prev = torch.zeros((B, self.latent_dim), device=x.device, dtype=x.dtype)

        # Start the autoregressive posterior calculation
        for t in range(T):

            # Get the context at time t
            h_t = h_rnn[:, t, :]

            # Concatenate it to the previous sample (along the last dimension)
            s_t = torch.cat([h_t, z_prev], dim=-1)

            # Apply the body module
            s_t = self.body_module(s_t)

            # The tentative z_t is computed using the transition module
            _, mu_trans, _ = self.transition_module.forward(z_prev)
            mu_enc, logvar = self.gaussian_module.get_params(s_t) 

            mu = mu_trans + mu_enc
            z = self.gaussian_module.sample(mu, logvar)

            # Save and go to the next iteration
            z_seq.append(z)
            mu_seq.append(mu)
            logvar_seq.append(logvar)

            z_prev = z  

        # Stack the lists to get the time dimension
        z_seq = torch.stack(z_seq, dim=1)
        mu_seq = torch.stack(mu_seq, dim=1)
        logvar_seq = torch.stack(logvar_seq, dim=1)

        # Unpack the leading dimensions
        z_seq = unpack_leading_dims(z_seq, input_shape)
        mu_seq = unpack_leading_dims(mu_seq, input_shape)
        logvar_seq = unpack_leading_dims(logvar_seq, input_shape)

        return z_seq, mu_seq, logvar_seq
    
    # Computes log q(z | x), but maybe it is trivial and I should put it in the forward
    # or make the "x" an argument (so that I can do the full computation instead of passing z)
    def log_prob(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

        return self.gaussian_module.log_prob(z, mu, logvar)


# ---------------------------------------------------------------------------- #
#                                   Decoders                                   #
# ---------------------------------------------------------------------------- #

class BaseDecoder(nn.Module, ABC):

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def log_prob(self) -> torch.Tensor:
        pass

    @abstractmethod
    def loss(self) -> torch.Tensor:
        pass


class RecurrentDiscreteDecoder(BaseDecoder):
    
    def __init__(self,
                 *,
                 recurrent_module: nn.Module,
                 body_module: Optional[nn.Module] = None,
                 num_classes: int):
        
        super().__init__()
        
        self.recurrent_module = recurrent_module
        self.body_module = body_module
        self.num_classes = num_classes
        self.decoder_head = nn.LazyLinear(self.num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        z, input_shape = pack_leading_dims(z)
        h, _ = self.recurrent_module(z)
        h = unpack_leading_dims(h, input_shape)

        if self.body_module is not None:
            h = self.body_module(h)
        y = self.decoder_head(h)

        return y
    
    # Assumes that, for discrete labels, the probability
    # is given by a softmax over the logits
    def log_prob(self, logits: torch.Tensor, y_true: torch.Tensor):

        # Compute the log-probabilities for all classes
        log_prob_all = F.log_softmax(logits, dim=-1)            

        # If needed, expand the true labels to the same shape as the predictions
        # Labels will be of shape (B, T, 1), so check if logits have an additional
        # leading dimension and expand
        y_true = standardize_shape(y_true)
        if logits.dim() == 4:
            y_true = y_true.unsqueeze(0).expand(logits.shape[0], -1, -1, -1) 

        # Select the log-prob corresponding to the real class
        log_p = log_prob_all.gather(dim=-1, index=y_true).squeeze(-1)

        return log_p

    def loss(self, logits: torch.Tensor, y_true: torch.Tensor):

        B, T, C = logits.shape

        # For cross entropy I need a 2D tensor for the predictions and 1D for the
        # true labels (assuming tensor of class indices)
        logits = logits.view(B * T, C)
        y_true = y_true.view(B * T)

        # Compute cross-entropy (sum over time, average over batch)
        return nn.functional.cross_entropy(logits, y_true) * T




class RecurrentContinuousDecoder(BaseDecoder):
    
    def __init__(self,
                 *,
                 recurrent_module: nn.Module,
                 body_module: Optional[nn.Module] = None,
                 output_dim: int
                 ):
        
        super().__init__()
        
        self.recurrent_module = recurrent_module
        self.body_module = body_module
        self.decoder_head = nn.LazyLinear(output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        z, input_shape = pack_leading_dims(z)
        h, _ = self.recurrent_module(z)
        h = unpack_leading_dims(h, input_shape)

        if self.body_module is not None:
            h = self.body_module(h)
        y = self.decoder_head(h)

        return y
    
    # For continuous data assume a simple gaussian with unitary variance
    # So an MSE loss
    def log_prob(self, y_pred: torch.Tensor, y_true: torch.Tensor):

        return -0.5 * (LOG2PI + (y_pred - y_true)**2 ).sum(dim=-1)

    # Simple MSE loss
    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        
        return ((y_pred - y_true)**2).sum(dim=-1).sum(dim=-1).mean()



class ResidualBlock(nn.Module):
    """
    A modern residual block with Layer Normalization.
    It uses a bottleneck structure to be more parameter-efficient.
    """
    def __init__(self, main_dim: int, expand_factor: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            # First layer expands the features
            nn.Linear(main_dim, main_dim * expand_factor),
            nn.LayerNorm(main_dim * expand_factor),
            nn.GELU(),
            # Second layer brings it back to the original dimension
            nn.Linear(main_dim * expand_factor, main_dim)
        )

    def forward(self, x):
        # The skip connection is the key!
        return x + self.block(x)
    
# ----------------------------------- ALTRO ---------------------------------- #
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


