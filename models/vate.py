import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List, Union
from os import PathLike
from models.layers import BaseEncoder, BaseDecoder, Transition

LOG2PI = math.log(2.0 * math.pi) 

class VATE(nn.Module):

    def __init__(
        self,
        encoder: BaseEncoder,
        transition: Transition,
        target_decoders: Dict[str, BaseDecoder],
    ):
        super().__init__()
        self.encoder = encoder
        self.transition = transition
        self.target_decoders = nn.ModuleDict(target_decoders)

    # Forward should just produce all the means, logvars, z samples, target predictions
    def forward(self, x: torch.Tensor, K: int = 1) -> Dict[str, Any]:
        
        # Encode the input data
        z_enc, mu_enc, logvar_enc = self.encoder(x)

        # One step transition
        z_trans, mu_trans, logvar_trans = self.transition(z_enc)
        
        # Decoder predictions (from samples, not averages)
        # TODO: Could use a mix of z_enc and z_trans if needed        
        target_preds: Dict[str, torch.Tensor] = {}
        for name, decoder in self.target_decoders.items():
            target_preds[name] = decoder(z_enc) 

        return {'z_enc': z_enc, 'mu_enc': mu_enc, 'logvar_enc': logvar_enc,
                'z_trans': z_trans, 'mu_trans': mu_trans, 'logvar_trans': logvar_trans,
                'target_preds': target_preds}
    

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
        # x = x.unsqueeze(0).expand(K, B, T, F)

        # Forward pass
        forward_output = self.forward(x, K=K)

        # Log-prob computation
        log_prob_output = self.get_log_probs(forward_output, y_targets)

        # Log_w terms of the loss
        log_w_prior, log_w_trans, log_w_dec = self.get_log_weights(log_prob_output)
        
        # print(log_prob_output['log_p'].max().item(), log_prob_output['log_q'].max().item())
        # print(log_w_trans.mean().item())

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
            losses = self.step_iw(batch, K=num_particles)
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
    
