import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List, Union
from os import PathLike


class VATE(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        transition: nn.Module,
        target_decoders: Dict[str, nn.Module],
        loss_fn_targets: Dict[str, nn.Module], 
    ):
        super().__init__()
        self.encoder = encoder
        self.transition = transition
        self.target_decoders = nn.ModuleDict(target_decoders)
        self.loss_fn_targets = nn.ModuleDict(loss_fn_targets)


    def forward(self, x: torch.Tensor, K: int = 1) -> Dict[str, Any]:

        B, T, F = x.shape # (batch_size, window_len, n_features)

        # Repeat input for K samples
        # TODO: check which one is faster (if they all give the same results)
        # x_tiled = x.repeat_interleave(K, dim=0) # (K*B,T,F)
        x_tiled = x.unsqueeze(0).expand(K, B, T, F)
        # x_tiled = x.unsqueeze(0).repeat(K, 1, 1, 1).contiguous()


        # Encode the input and get the latent samples and corresponding log-prob
        # Note that it's not possible to encode x just once and then use the mean/logvar
        # to sample K times, as the encoder is supposed to be autoregressive/recurrent so the
        # each mu/logvar dependes on the sampled point at the previous timestep z_{t-1}
        z_enc, log_q = self.encoder(x_tiled)  # (K,B,T,Z), (K,B,T)

        # Get the prior transition probabilities
        z_trans, log_p = self.transition(z_enc)     # (K,B,T-1)

        # Get the prior at the first timestep (assume standard gaussian)
        # log_p0 = -(torch.sum(z_enc[:, :, 0, :]**2, dim=-1) + math.log(2 * math.pi)) / 2  # (K,B)
        log_p0 = -0.5 * (torch.sum(z_enc[:, :, 0, :]**2, dim=-1) + z_enc.size(-1) * math.log(2 * math.pi))
        
        # z_mixed = torch.zeros_like(z_enc)
        # z_mixed[:, :, 1:, :] = z_trans[:, :, :-1, :]
        
        # Decoder predictions per sample        
        target_preds: Dict[str, torch.Tensor] = {}
        for name, decoder in self.target_decoders.items():
            target_preds[name] = decoder(z_enc) 

        return {'z_enc': z_enc, 'log_p': log_p, 'log_q': log_q, 'log_p0': log_p0, 'target_preds': target_preds}
    

    # Compute the log-prob of the targets labels
    def compute_decoder_logprob(self, targets: Dict[str, torch.Tensor], target_preds: Dict[str, torch.Tensor]):

        # Infer K, B from any head
        first_pred = next(iter(target_preds.values()))
        K, B, T = first_pred.shape[:3]
        log_p_decoders = dict()

        # Cycle through each head (different target labels)
        for name, preds_k in target_preds.items():

            loss_fn = self.loss_fn_targets[name]

            # ----- Multiclass classification: CrossEntropyLoss -----
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                # preds_k: (K,B,T,C), targets[name]: (B,T) with class indices
                logits = preds_k
                log_probs = F.log_softmax(logits, dim=-1)             # (K,B,T,C)
                y = targets[name].unsqueeze(0).expand(K, -1, -1, -1)  # (K,B,T,1)
                logp = log_probs.gather(dim=-1, index=y).squeeze(-1)  # (K,B,T)
                logp_total = logp.sum(dim=2)                                 

            # # ----- Regression (fixed var): MSELoss -> Gaussian with σ^2 = 1 (const dropped) -----
            # elif isinstance(loss_fn, nn.MSELoss):
            #     # preds_k: mean (K,B,T,...) , targets[name]: (B,T,...)
            #     mu = preds_k
            #     y = targets[name].unsqueeze(0).expand_as(mu)
            #     se = (mu - y) ** 2
            #     sum_dims = tuple(range(2, mu.dim()))                         # sum over time and features
            #     logp_total = - 0.5 * se.sum(dim=sum_dims)

            # # ----- Regression (heteroscedastic): GaussianNLLLoss -> provide (mu, logvar) -----
            # elif isinstance(loss_fn, nn.GaussianNLLLoss):
            #     # preds_k: tuple (mu, logvar), each (K,B,T,...) matching targets
            #     mu, logvar = preds_k  # expect a tuple from your decoder
            #     y = targets[name].unsqueeze(0).expand_as(mu)
            #     inv_var = torch.exp(-logvar)
            #     # log N(y | mu, var) up to additive constant
            #     logp_elem = -0.5 * ((y - mu) ** 2 * inv_var + logvar)
            #     sum_dims = tuple(range(2, mu.dim()))
            #     logp_total = logp_elem.sum(dim=sum_dims)

            # # Optional: L1Loss as Laplace(μ, b=1) up to const
            # elif isinstance(loss_fn, nn.L1Loss):
            #     mu = preds_k
            #     y = targets[name].unsqueeze(0).expand_as(mu)
            #     abs_err = (mu - y).abs()
            #     sum_dims = tuple(range(2, mu.dim()))
            #     logp_total = -abs_err.sum(dim=sum_dims)      

            log_p_decoders[name] = logp_total          
                                                
        return log_p_decoders


    def compute_log_weights(self, batch: Dict[str, Any], K: int = 1):

        # Compute log(w) = log(p(y,z)/q(z|x))
        x = batch['neural']
        y_targets = batch['targets']
        forward_outs = self.forward(x, K=K)

        log_p = forward_outs['log_p'] 
        log_q = forward_outs['log_q']
        log_p0 = forward_outs['log_p0']
        target_preds = forward_outs['target_preds']

        # Prior term
        log_w_prior = log_p0 - log_q[:, :, 0] 

        # Transition terms
        # log_w_trans = (log_p - log_q[:, :, 1:]).sum(dim=2) 
        log_w_trans = (log_p - log_q[:, :, 1:].detach()).sum(dim=2) 

        # Decoder terms
        log_w_dec = self.compute_decoder_logprob(y_targets, target_preds)

        return log_w_prior, log_w_trans, log_w_dec

    # def loss(self, log_w_prior, log_w_trans, log_w_dec):

    #     # Compute log(w)
    #     log_w = log_w_prior + log_w_trans + sum(log_w_dec.values())

    #     # Subtract max trick for numerical stability
    #     m = log_w.max(dim=0, keepdim=True).values
    #     loss = -(torch.logsumexp(log_w - m, dim=0) + m.squeeze(0) - math.log(log_w.shape[0])).mean()
    #     return loss

    # def step(self, batch: Dict[str, Any], K: int = 1) -> Dict[str, Any]:
        
    #     # Compute the log_w terms (forward pass)
    #     log_w_terms = self.compute_log_weights(batch, K)

    #     # Compute the IW loss (all terms added together)
    #     loss = self.loss(*log_w_terms)

    #     # Compute the importance-weighted terms for debug and visualization


    #     losses = {'total_loss': loss}

    #     return losses


    def step(self, batch: Dict[str, Any], K: int = 1) -> Dict[str, Any]:

        # Forward pass terms
        log_w_prior, log_w_trans, log_w_dec = self.compute_log_weights(batch, K)

        # Compute S = log(w)
        S = log_w_prior + log_w_trans
        for v in log_w_dec.values():
            S = S + v

        # Subtract max trick for numerical stability
        S_max = S.max(dim=0, keepdim=True).values       # (1,B)
        S_centered = S - S_max                          # (K,B)

        # Compute IW-ELBO and loss
        logsum = torch.logsumexp(S_centered, dim=0)     # (B,)
        iw_elbo = logsum + S_max.squeeze(0) - math.log(S.size(0))
        loss = -iw_elbo.mean()

        # Diagnostics and visualizations
        with torch.no_grad():

            # Compute weights
            w_tilde = torch.exp(S_centered - logsum.unsqueeze(0))  # (K,B)

            # Compute the contribution of each term to the IW-ELBO
            iw_init = (w_tilde * log_w_prior).sum(dim=0).mean()
            iw_tr   = (w_tilde * log_w_trans).sum(dim=0).mean()
            iw_dec  = {name: (w_tilde * v).sum(dim=0).mean() for name, v in log_w_dec.items()}

            # Entropy correction term
            logZ = torch.logsumexp(S_centered, dim=0, keepdim=True)   # (1,B)
            H = -(w_tilde * (S_centered - logZ)).sum(dim=0)           # (B,)
            entropy_term = (H - math.log(w_tilde.size(0))).mean()

        losses = {
            'total_loss': loss,
            'entropy': -entropy_term.detach(),
            'loss_prior': -iw_init.detach(),
            'loss_transition': -iw_tr.detach(),
            'loss_targets': {n: -v.detach() for n, v in iw_dec.items()}
        }
        return losses


    def fit(self,
            dataset,
            iterations: int,
            batch_size: int = 128,
            window_len: int = 2,
            num_particles: int = 5,
            optimizer: Optional[Optimizer] = None,
            callbacks: Optional[List[Any]] = None,
            show_progress: bool = True
            ) -> Dict[str, List[float]]:
    
        self.train()

        # Optimizer ----------------------------------------------------------------
        if optimizer is None:
            optimizer = Adam(self.parameters(), lr=3e-4)

        # Loss history -------------------------------------------------------------
        history = {'total_loss': [], 'loss_transition': [], 'loss_prior': [], 'entropy': []}
        history.update({f'loss_targets_{n}': [] for n in self.target_decoders})

        max_start = dataset.T - window_len
        if max_start < 0:
            raise ValueError(f'window_len={window_len} is longer than the sequence ({dataset.T}).')

        pbar = tqdm(range(iterations), disable=not show_progress)

        for it in pbar:

            # Sample a random batch
            idxs = torch.randint(low=0, high=max_start + 1, size=(batch_size,), 
                                 device=dataset.device, dtype=torch.long)
            
            batch = dataset.gather(idxs, window_len)
            
            # Optimization step
            optimizer.zero_grad()
            losses = self.step(batch, K=num_particles)
            losses['total_loss'].backward()
            optimizer.step()

            # Track losses
            history['total_loss'].append(losses['total_loss'].item())
            history['loss_transition'].append(losses['loss_transition'].item())
            history['loss_prior'].append(losses['loss_prior'].item())
            history['entropy'].append(losses['entropy'].item())

            for n, l in losses['loss_targets'].items():
                history[f'loss_targets_{n}'].append(l.item())
            # for n, l in losses['loss_nuisances'].items():
            #     history[f'loss_nuisances_{n}'].append(l.item())

            # Progress bar text
            if show_progress:
                pbar.set_postfix({k: v[-1] for k, v in history.items()})

        return history
    
    
    def generate(self,
                 *,                       # keyword-only API
                 num_steps: int,
                 x0: Optional[torch.Tensor] = None,
                 z0: Optional[torch.Tensor] = None,
                 ) -> Dict[str, Any]:

        self.eval()

        with torch.no_grad():
            # If the starting point is the neural data, encode it first
            if z0 is None:
                if x0 is None:
                    raise ValueError("generate: provide either x0 or z0")
                x0 = x0.view(1,1,1,-1)
                z_current, _ = self.encoder(x0)
            else:
                z_current = z0

            # Adjust the shape 
            if z_current.ndim == 2:
                z_current = z_current[-1,:]
            elif z_current.ndim == 3: 
                z_current = z_current[-1,-1,:]

            # Rollout
            zs = [z_current]
            
            for _ in range(num_steps):
                z_next, _ = self.transition(z_current)
                # mu, logvar = self.transition.get_gaussian_params(z_current)
                # std = torch.exp(0.5 * logvar)
                # eps = torch.randn_like(std)
                # z_next = mu + std * eps * 2

                zs.append(z_next)
                z_current = z_next

            z_sequence = torch.cat(zs, dim=2).squeeze()                         # (num_steps+1, Z)

            # Decode targets
            target_preds = {
                n: dec(z_sequence.unsqueeze(0)).squeeze(0)            # (num_steps+1, …)
                for n, dec in self.target_decoders.items()
            }

        return {'z_sequence': z_sequence,
                'target_preds': target_preds}


    def summary(self, input_features: int):

        device = next(self.parameters()).device

        dummy_x = torch.zeros((1,1,5,input_features), dtype=torch.float32, device=device)
        dummy_z, _ = self.encoder(dummy_x)
        _ = self.transition(dummy_z)

        encoder_param_count = sum(p.numel() for p in self.encoder.parameters()) 
        transition_param_count = sum(p.numel() for p in self.transition.parameters()) 

        print(f'{"Module":<{15}}{"Parameters":<{15}}')
        print(f'{"-"*30:<{30}}')
        print(f'{"Encoder":<{15}}{encoder_param_count:<{15}}')
        print(f'{"Transition":<{15}}{transition_param_count:<{15}}')

        dec_params_count = {}

        for name in self.target_decoders:
            dummy_y_preds = self.target_decoders[name](dummy_z)
            decoder_param_count = sum(p.numel() for p in  self.target_decoders[name].parameters()) 
            dec_params_count[name] = decoder_param_count
           
        if len(dec_params_count) > 1:
            print(f'{"Decoders":<{15}}{sum(dec_params_count.values()):<{15}}')
        else:
            print(f'{"Decoders":<{15}}')

        for name in dec_params_count:
            print(f'{"-" + name:<{15}}{dec_params_count[name]:<{15}}')

        total_params = encoder_param_count + transition_param_count + sum(dec_params_count.values())

        print(f'{"-"*30:<{30}}')
        print(f'{"Total":<{15}}{total_params:<{15}}')


    def save(self, path: Union[str, PathLike]):
        torch.save(self.state_dict(), path)

    def load(self, path: Union[str, PathLike], map_location: str = "cpu"):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        return self
    
# ---------------------------------------------------------------------------- #
#                                    Modules                                   #
# ---------------------------------------------------------------------------- #
def diag_normal_log_prob(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (
        (x - mu).pow(2) * torch.exp(-logvar) + 
        logvar + torch.log(torch.tensor(2.0 * torch.pi, device=x.device))
    ).sum(dim=-1)


class RecurrentEncoder(nn.Module):

    def __init__(
        self,
        *,
        rnn_module: nn.Module,
        body_module: nn.Module,
        latent_dim: int,
        logvar_lims: Tuple[float, float] = (-9.0, 2.0),
        fixed_logvar: Optional[float] = None,   # if not None, use this instead of a logvar head
    ):
        super().__init__()
        self.rnn_module = rnn_module
        self.body_module = body_module
        self.latent_dim = int(latent_dim)
        self.logvar_lims = logvar_lims

        # Heads are inferred lazily from body_module output size
        self.mean_layer = nn.LazyLinear(self.latent_dim)

        if fixed_logvar is None:
            self.logvar_layer = nn.LazyLinear(self.latent_dim)
            self.register_buffer("logvar_const_buf", None, persistent=False)
        else:
            self.logvar_layer = None
            self.register_buffer(
                "logvar_const_buf",
                torch.full((1, self.latent_dim), float(fixed_logvar))
            )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4, "Encoder expects (K, B, T, F). Reshape in VATE before calling."
        K, B, T, F = x.shape
        KB = K * B

        # RNN expects 3D -> flatten particles & batch (view, no copy if contiguous)
        x_flat = x.reshape(KB, T, F)                       # (K*B, T, F)
        h_seq, _ = self.rnn_module(x_flat)                 # (K*B, T, H)

        # Autoregressive posterior calculation
        # Start from z = 0 at t = -1 and autoregressively expand it
        z_prev = torch.zeros(KB, self.latent_dim, device=x.device)
        z_seq = []
        log_q_seq = []

        for t in range(T):
            
            # Get the context from the neural data and concatenate it with the previous z
            h = h_seq[:, t, :]                         
            s = torch.cat([h, z_prev], dim=-1)       

            # Apply some non-linear layers
            f = self.body_module(s)                    

            # Get the mean and logvar for q(z_t | ...)
            mu = self.mean_layer(f)                    

            if self.logvar_layer is not None:
                logvar = self.logvar_layer(f)    
            else:
                logvar = self.logvar_const_buf.expand_as(mu)
            logvar = torch.clamp(logvar, *self.logvar_lims)

            # Sample with the reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps                     

            # Calculate the probability of this sample
            log_q = diag_normal_log_prob(z, mu, logvar)

            # Append to the sequences and go to the next step
            z_seq.append(z)
            log_q_seq.append(log_q)

            z_prev = z

        # Stack the sequences (to recover the time dimension) and reshape to separate K and B
        z_seq = torch.stack(z_seq, dim=1).reshape(K, B, T, self.latent_dim)    # (K,B,T,Z)
        log_q_seq = torch.stack(log_q_seq, dim=1).reshape(K, B, T)              # (K,B,T)

        return z_seq, log_q_seq


class TransitionModule(nn.Module):
    """
    IWAE-style transition:
      forward(z_enc) -> z_next_samples, logp_tr
        z_next_samples : (..., T-1, Z) sampled from p(z_t | z_{t-1})
        logp_tr        : (..., T-1)    log p(z_t | z_{t-1}) evaluated at the provided z_enc

    Residual modes:
      - "plain"    : mu = v
      - "residual" : mu = z_prev + v
      - "tanh"     : mu = z_prev + residual_scale * tanh(v)
      - "unit_step": mu = z_prev + residual_scale * normalize(v)
    """
    def __init__(
        self,
        *,
        base_module: nn.Module,            # maps (..., Z) -> (..., H)
        latent_dim: int,
        residual_mode: str = "tanh",
        residual_scale: float = 0.05,
        logvar_lims: Tuple[float, float] = (-9.0, 2.0),
        fixed_logvar: Optional[float] = None,   # if set, fixes transition variance
    ):
        super().__init__()
        self.base_module = base_module
        self.latent_dim = int(latent_dim)
        self.residual_mode = residual_mode
        self.residual_scale = float(residual_scale)
        self.logvar_lims = logvar_lims

        # Heads inferred lazily from base_module output
        self.mean_layer = nn.LazyLinear(self.latent_dim)
        if fixed_logvar is None:
            self.logvar_layer = nn.LazyLinear(self.latent_dim)
            self.register_buffer("logvar_const_buf", None, persistent=False)
        else:
            self.logvar_layer = None
            self.register_buffer("logvar_const_buf", torch.full((1, self.latent_dim), float(fixed_logvar)))

    def get_gaussian_params(self, z_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        flat = z_prev.reshape(-1, z_prev.size(-1))      
        h = self.base_module(flat)                 
        v = self.mean_layer(h).reshape_as(z_prev)           

        if self.residual_mode == "plain":
            mu = v
        elif self.residual_mode == "residual":
            mu = z_prev + v
        elif self.residual_mode == "tanh":
            mu = z_prev + self.residual_scale * torch.tanh(v)
        elif self.residual_mode == "unit_step":
            u = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
            mu = z_prev + self.residual_scale * u

        if self.logvar_layer is not None:
            logvar = self.logvar_layer(h).reshape_as(z_prev)
        else:
            logvar = self.logvar_const_buf.expand_as(mu)

        logvar = torch.clamp(logvar, *self.logvar_lims)
        return mu, logvar



    def forward(self, z_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert z_enc.dim() in (3, 4)

        # Get the parameters (for all timesteps)
        mu, logvar = self.get_gaussian_params(z_enc)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_next = mu + std * eps      

        # log p(z_t | z_{t-1}) evaluated at provided z_curr / z_prev
        log_p_tr = diag_normal_log_prob(z_enc[:, :, 1:, :], mu[:, :, :-1, :], logvar[:, :, :-1, :])  # (..., T-1)
        
        return z_next, log_p_tr


    # def forward(self, z_enc: torch.Tensor) -> torch.Tensor:

    #     assert z_enc.dim() in (3, 4)
    #     if z_enc.dim() == 3:
    #         # (B, T, Z)
    #         z_prev = z_enc[:, :-1, :]                       # (B, T-1, Z)
    #         z_curr = z_enc[:,  1:, :]                       # (B, T-1, Z)
    #     else:
    #         # (K, B, T, Z)
    #         z_prev = z_enc[:, :, :-1, :]                    # (K, B, T-1, Z)
    #         z_curr = z_enc[:, :,  1:, :]                    # (K, B, T-1, Z)

    #     # Get the parameters (for all timesteps)
    #     mu, logvar = self.get_gaussian_params(z_prev)

    #     # log p(z_t | z_{t-1}) evaluated at provided z_curr / z_prev
    #     log_p_tr = diag_normal_log_prob(z_curr, mu, logvar)  # (..., T-1)
    #     return log_p_tr


    # def step(self, z: torch.Tensor) -> torch.Tensor:

    #     mu, logvar = self.get_gaussian_params(z)

    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     z_next = mu + std * eps                    

    #     return z_next
    

class RecurrentDecoder(nn.Module):

    def __init__(self, 
                 rnn_module: nn.Module,
                 decoder_module: nn.Module):
        
        super().__init__()
        self.rnn_module = rnn_module
        self.decoder_module = decoder_module

    def forward(self, x):
        
        x_ndim = x.dim() 

        if x_ndim == 4:
            K, B = x.shape[0], x.shape[1]
            x = x.flatten(0,1)

        x, _ = self.rnn_module(x)
        x = self.decoder_module(x)

        if x_ndim == 4:
            x = x.reshape(K, B, *x.shape[1:])

        return x
    