import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List, Union, Mapping
from os import PathLike
from models.layers import BaseEncoder, BaseDecoder, Transition

LOG2PI = math.log(2.0 * math.pi) 

def _kl_div_standard(mu, logvar):
    return 0.5 * (logvar.exp() + mu**2 - logvar - 1)

def _kl_div(mu_1, mu_2, logvar_1, logvar_2):
    return 0.5 * ((logvar_1 - logvar_2).exp() + (mu_1 - mu_2)**2 / logvar_2.exp() +  logvar_2 - logvar_1 - 1)


class VATE(nn.Module):

    def __init__(
        self,
        encoder: BaseEncoder,
        transition: Transition,
        target_decoders: Mapping[str, BaseDecoder],
        decode_from_posterior: bool = True
    ):
        super().__init__()
        self.encoder = encoder
        self.transition = transition
        self.target_decoders = nn.ModuleDict(target_decoders)

        self.decode_from_posterior = decode_from_posterior

    # Forward should just produce all the means, logvars, z samples, target predictions
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        
        # Encode the input data
        z_enc, mu_enc, logvar_enc = self.encoder(x)

        # One step transition
        z_trans, mu_trans, logvar_trans = self.transition(z_enc)
        
        # Decoder predictions (from samples, not averages)
        # TODO: Could use a mix of z_enc and z_trans if needed        
        z_to_decode = z_enc
        if not self.decode_from_posterior:
            z_to_decode = torch.cat((z_enc[...,0:1,:], z_trans[..., :-1,:]), dim=-2)

        # z_to_decode = mu_enc # NOTE: REMOVE OR INTEGRATE BETTER

        target_preds: Dict[str, torch.Tensor] = {}
        for name, decoder in self.target_decoders.items():
            target_preds[name] = decoder(z_to_decode) 

        return {'z_enc': z_enc, 'mu_enc': mu_enc, 'logvar_enc': logvar_enc,
                'z_trans': z_trans, 'mu_trans': mu_trans, 'logvar_trans': logvar_trans,
                'target_preds': target_preds}
    

    # ---------------------------------- IW-ELBO --------------------------------- #

    # Used in IW-ELBO
    def get_log_probs(self, forward_output: Dict[str, Any], target_labels: torch.Tensor):

        z_enc, mu_enc, logvar_enc = forward_output['z_enc'], forward_output['mu_enc'], forward_output['logvar_enc']
        z_trans, mu_trans, logvar_trans = forward_output['z_trans'], forward_output['mu_trans'], forward_output['logvar_trans']
        target_preds = forward_output['target_preds']

        # Compute log_p(z_t | z_{t-1}) and log_q(z_t | z_{t-1}, x_{0:t})
        log_q = self.encoder.log_prob(z_enc, mu_enc, logvar_enc)
        log_p = self.transition.log_prob(z_enc[..., 1:, :],
                                         mu_trans[..., :-1, :],
                                         logvar_trans[..., :-1, :])

        # Isolate t=0 and remove the prior sample at t+1
        log_q_0 = log_q[..., 0]
        log_q = log_q[..., 1:]

        # Assume the p_0 is just a standard gaussian
        log_p_0 = -0.5 * (torch.sum(z_enc[..., 0, :]**2, dim=-1) + z_enc.shape[-1] * LOG2PI)

        # Compute the label-decoders log-p
        log_p_decoders = dict()

        # Cycle through each head (different target labels)
        for name, pred in target_preds.items():
            decoder = self.target_decoders[name]
            true_label = target_labels[name]
            log_p_decoders[name] = decoder.log_prob(pred, true_label) # type: ignore
        
        return {'log_p_0': log_p_0, 'log_q_0': log_q_0,
                'log_p': log_p, 'log_q': log_q,
                'log_p_decoders': log_p_decoders}
        

    def get_log_weights(self, log_prob_output: Dict[str, Any]):

        log_p_0, log_q_0 = log_prob_output['log_p_0'], log_prob_output['log_q_0']
        log_p, log_q = log_prob_output['log_p'], log_prob_output['log_q']

        # Prior term
        log_w_prior = log_p_0 - log_q_0

        # Transition terms (summed over time)
        # TODO: give the possibility to detach either term (or mix the two detached
        # terms to get a weighted gradient.
        log_w_trans = (log_p - log_q).sum(dim=-1) 

        # Decoder terms (summed over time)
        log_w_dec = {name: log_p_dec.sum(dim=-1) for name, log_p_dec in log_prob_output['log_p_decoders'].items()}
        
        return log_w_prior, log_w_trans, log_w_dec


    def step_iw(self, batch: Dict[str, Any], K: int = 1) -> Dict[str, Any]:

        x = batch['neural']
        y_targets = batch['targets']

        x = x.unsqueeze(0).repeat(K, 1, 1, 1).contiguous()

        # Forward pass
        forward_output = self.forward(x)

        # Log-prob computation
        log_prob_output = self.get_log_probs(forward_output, y_targets)

        # Log_w terms of the loss
        log_w_prior, log_w_trans, log_w_dec = self.get_log_weights(log_prob_output)
        
        # Compute S = log(w)
        S = log_w_prior + log_w_trans * 10
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

    # ------------------------------- Standard ELBO ------------------------------ #
    def step_elbo(self, batch: Dict[str, Any], kl_weight: float = 1) -> Dict[str, Any]:

        x = batch['neural']
        y_targets = batch['targets']

        # Forward pass
        forward_output = self.forward(x)

        z_enc, mu_enc, logvar_enc = forward_output['z_enc'], forward_output['mu_enc'], forward_output['logvar_enc']
        z_trans, mu_trans, logvar_trans = forward_output['z_trans'], forward_output['mu_trans'], forward_output['logvar_trans']
        target_preds = forward_output['target_preds']
        
        # Latent prior loss
        mu_0, logvar_0 = mu_enc[:, 0], logvar_enc[:, 0]
        # loss_prior = _kl_div_standard(mu_0, logvar_0).sum(dim=-1).mean()
        
        # # Transition loss 
        # loss_transition = _kl_div(mu_enc[:, 1:, :], mu_trans[:, :-1, :],
        #                           logvar_enc[:, 1:, :], logvar_trans[:, :-1, :]).sum(dim=-1).sum(dim=-1).mean()

        kl_div_prior = _kl_div_standard(mu_0, logvar_0)
        kl_div_trans = _kl_div(mu_enc[:, 1:, :], mu_trans[:, :-1, :], logvar_enc[:, 1:, :], logvar_trans[:, :-1, :])
        
        free_bits_per_dim = 0.1
        loss_prior = torch.clamp(kl_div_prior, min=free_bits_per_dim).sum(dim=-1).mean()
        loss_transition = torch.clamp(kl_div_trans, min=free_bits_per_dim).sum(dim=-1).sum(dim=-1).mean()

        # Decoder losses
        # TODO: spostarlo dentro la classe del decoder 
        loss_targets = {}
        for name, pred in target_preds.items():
            
            target_label_true = y_targets[name]

            # Pack leading dims for the corss-entropy
            B, T, C = pred.shape
            pred = pred.reshape(B * T, C)
            target_label_true = target_label_true.reshape(B * T)
            loss_targets[name] = nn.functional.cross_entropy(pred, target_label_true) * x.shape[1]
   
        kl_weight = math.exp(-10)
        total_loss = (loss_prior + loss_transition) * kl_weight + sum(loss_targets.values())

        return {
            'total_loss': total_loss,
            'loss_prior': loss_prior,
            'loss_transition': loss_transition,
            'loss_targets': loss_targets,
        }
    

    # This is the basic, non-variational, approach
    def step_basic(self, batch: Dict[str, Any]) -> Dict[str, Any]:

        x = batch['neural']
        y_targets = batch['targets']

        # Forward pass
        forward_output = self.forward(x)
        z_enc = forward_output['z_enc']
        z_trans = forward_output['z_trans']
        target_preds = forward_output['target_preds']
        mu_enc = forward_output['mu_enc']
        mu_trans = forward_output['mu_trans']

        # loss_transition = ((z_enc[:, 1:, :] - z_trans[:, :-1, :])**2).sum(dim=-1).sum(dim=-1).mean()
        loss_transition = ((mu_enc[:, 1:, :] - mu_trans[:, :-1, :])**2).sum(dim=-1).sum(dim=-1).mean()

        loss_targets = {}
        for name, pred in target_preds.items():
            
            target_label_true = y_targets[name]

            # Pack leading dims for the corss-entropy
            B, T, C = pred.shape
            pred = pred.reshape(B * T, C)
            target_label_true = target_label_true.reshape(B * T)
            loss_targets[name] = nn.functional.cross_entropy(pred, target_label_true) * T
   
        total_loss = loss_transition * 10 + sum(loss_targets.values())

        return {
            'total_loss': total_loss,
            'loss_transition': loss_transition,
            'loss_targets': loss_targets,
        }
    

    # ------------------------- Balanced ELBO (EXPERIMENTAL) ------------------------- #
    def _get_grad_norm(self, loss: torch.Tensor, params: Any) -> torch.Tensor:
        """Computes the L2 norm of gradients of a loss with respect to parameters."""
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        # Filter out None gradients for layers that are not affected
        valid_grads = [g for g in grads if g is not None]
        if not valid_grads:
            return torch.tensor(0.0, device=loss.device)
        # Flatten all gradients into a single vector and compute the L2 norm
        grad_norm = torch.cat([g.view(-1) for g in valid_grads]).norm(2)
        return grad_norm
    
    
    def step_elbo_balanced(self, batch: Dict[str, Any], kl_beta: float = 1.0) -> Dict[str, Any]:

        x = batch['neural']
        y_targets = batch['targets']

        # Forward pass to get all model outputs
        forward_output = self.forward(x)

        mu_enc, logvar_enc = forward_output['mu_enc'], forward_output['logvar_enc']
        mu_trans, logvar_trans = forward_output['mu_trans'], forward_output['logvar_trans']
        target_preds = forward_output['target_preds']
        
        # -------------------- 1. Calculate Individual Losses -------------------- #
        
        # --- KL Loss (Rate) ---
        mu_0, logvar_0 = mu_enc[:, 0], logvar_enc[:, 0]
        loss_prior = _kl_div_standard(mu_0, logvar_0).sum(dim=-1).mean()
        
        loss_transition = _kl_div(mu_enc[:, 1:, :], mu_trans[:, :-1, :],
                                  logvar_enc[:, 1:, :], logvar_trans[:, :-1, :]).sum(dim=-1).sum(dim=-1).mean()

        # Combine KL terms (using your 10x weight for transition)
        loss_kl = loss_prior + loss_transition
        
        # --- Reconstruction Loss (Distortion) ---
        loss_targets = {}
        for name, pred in target_preds.items():
            target_label_true = y_targets[name]
            B, T, C = pred.shape
            pred_flat = pred.view(B * T, C)
            target_flat = target_label_true.view(B * T)
            loss_targets[name] = nn.functional.cross_entropy(pred_flat, target_flat) * T
        
        loss_recon = sum(loss_targets.values())

        # -------------------- 2. Compute Gradient Balance (λ) ------------------- #

        # The encoder parameters are the "shared" weights influenced by both losses
        shared_params = list(self.encoder.parameters())

        # Get the norm of the reconstruction gradient
        norm_recon = self._get_grad_norm(loss_recon, shared_params)
        
        # Get the norm of the KL gradient
        norm_kl = self._get_grad_norm(loss_kl, shared_params)
        
        # Calculate the balancing factor λ
        # We add a small epsilon to prevent division by zero
        # The .detach() is CRUCIAL: λ is treated as a constant, not a parameter
        lamb = (norm_recon / (norm_kl + 1e-6)).detach()

        # We can also apply the overall beta factor from beta-VAE here
        lamb = kl_beta * lamb
        
        # ------------------------ 3. Compute Final Loss ------------------------- #
        
        total_loss = loss_recon + lamb * loss_kl

        return {
            'total_loss': total_loss,
            'loss_prior': loss_prior.detach(),
            'loss_transition': loss_transition.detach(),
            'loss_targets': {k: v.detach() for k, v in loss_targets.items()},
            'lambda': lamb.detach() # Monitor this value during training!
        }
    


    def fit(self,
            dataset,
            iterations: int,
            batch_size: int = 128,
            window_len: int = 2,
            num_particles: int = 5,
            optimizer: Optional[Optimizer] = None,
            variational_loss: bool = True,
            show_progress: bool = True
            ) -> Dict[str, List[float]]:

        # Set the model in training mode
        self.train()

        # Instantiate an optimizer if not provided 
        if optimizer is None:
            optimizer = Adam(self.parameters(), lr=3e-4)

        # Check if the dataset is long enough for the training
        max_start = dataset.T - window_len
        if max_start < 0:
            raise ValueError(f'window_len={window_len} is longer than the sequence ({dataset.T}).')

        # Loss history -------------------------------------------------------------
        # history = {'total_loss': [], 'loss_transition': [], 'loss_prior': [], 'entropy': []}
        # history.update({f'loss_targets_{n}': [] for n in self.target_decoders})
        history = dict()

        # Initialize the progress bar and start training
        pbar = tqdm(range(iterations), disable=not show_progress)

        for it in pbar:

            kl_weight = min(1.0, it / (iterations * 0.5)) 

            # Sample a random batch
            idxs = torch.randint(low=0, high=max_start + 1, size=(batch_size,), 
                                 device=dataset.device, dtype=torch.long)
            
            batch = dataset.gather(idxs, window_len)
            
            # Optimization step
            optimizer.zero_grad()

            if variational_loss:
                if num_particles == 1:
                    losses = self.step_elbo(batch, kl_weight=kl_weight)
                    # losses = self.step_elbo_balanced(batch, kl_beta=1)
                else:
                    losses = self.step_iw(batch, K=num_particles)
            else:
                losses = self.step_basic(batch)

            losses['total_loss'].backward()
            optimizer.step()

            # Statistics tracking
            
            # At first iteration initialize a list for each stat and handle the nested lists
            # Then add the corresponding stat value for the current iteration
            for stat_name, stat_value in losses.items():
                if isinstance(stat_value, torch.Tensor):
                    if stat_name not in history:
                        history[stat_name] = []
                    history[stat_name].append(stat_value.item())
                elif isinstance(stat_value, dict):
                    for key in stat_value:
                        key_name = stat_name + '_' + key
                        if key_name not in history:
                            history[key_name] = []
                        history[key_name].append(stat_value[key].item())

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
                z_current, _, _ = self.encoder(x0)
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
                z_next, _, _ = self.transition(z_current)
                zs.append(z_next)
                z_current = z_next

            z_sequence = torch.cat(zs, dim=-2).squeeze()                         # (num_steps+1, Z)

            # Decode targets
            target_preds = {
                n: dec(z_sequence.unsqueeze(0)).squeeze(0)            # (num_steps+1, …)
                for n, dec in self.target_decoders.items()
            }

        return {'z_sequence': z_sequence,
                'target_preds': target_preds}


    def summary(self, input_features: int):

        device = next(self.parameters()).device

        dummy_x = torch.zeros((1,1,input_features), dtype=torch.float32, device=device)
        dummy_z, _, _ = self.encoder(dummy_x)
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
    
