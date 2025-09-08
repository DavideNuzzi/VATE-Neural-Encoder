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
        
        target_preds: Dict[str, torch.Tensor] = {}
        for name, decoder in self.target_decoders.items():
            target_preds[name] = decoder(z_to_decode) 

        return {'z_enc': z_enc, 'mu_enc': mu_enc, 'logvar_enc': logvar_enc,
                'z_trans': z_trans, 'mu_trans': mu_trans, 'logvar_trans': logvar_trans,
                'target_preds': target_preds}


    # ------------------------------- Standard ELBO ------------------------------ #
    def step_elbo(self, batch: Dict[str, Any]) -> Dict[str, Any]:

        x = batch['neural']
        y_targets = batch['targets']

        # Forward pass
        forward_output = self.forward(x)

        z_enc, mu_enc, logvar_enc = forward_output['z_enc'], forward_output['mu_enc'], forward_output['logvar_enc']
        z_trans, mu_trans, logvar_trans = forward_output['z_trans'], forward_output['mu_trans'], forward_output['logvar_trans']
        target_preds = forward_output['target_preds']
        
        # Latent prior loss
        mu_0, logvar_0 = mu_enc[:, 0], logvar_enc[:, 0]
        loss_prior = _kl_div_standard(mu_0, logvar_0).sum(dim=-1).mean()
        
        # # Transition loss 
        loss_transition = _kl_div(mu_enc[:, 1:, :], mu_trans[:, :-1, :],
                                  logvar_enc[:, 1:, :], logvar_trans[:, :-1, :]).sum(dim=-1).sum(dim=-1).mean()

        # Decoder losses
        loss_targets = {}
        for name, pred in target_preds.items():
            
            target_label_true = y_targets[name]
            decoder = self.target_decoders[name] 
            loss_targets[name] = decoder.loss(pred, target_label_true) # type: ignore

        kl_weight = math.exp(-10)
        total_loss = (loss_prior + loss_transition) * kl_weight + sum(loss_targets.values())

        return {
            'total_loss': total_loss,
            'loss_prior': loss_prior,
            'loss_transition': loss_transition,
            'loss_targets': loss_targets,
        }
    

    def step_elbo

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
            decoder = self.target_decoders[name] 
            loss_targets[name] = decoder.loss(pred, target_label_true) # type: ignore

        total_loss = loss_transition * 10 + sum(loss_targets.values())

        return {
            'total_loss': total_loss,
            'loss_transition': loss_transition,
            'loss_targets': loss_targets,
        }
    


    def fit(self,
            dataset,
            iterations: int,
            batch_size: int = 128,
            window_len: int = 2,
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

        # Loss history
        history = dict()

        # Initialize the progress bar and start training
        pbar = tqdm(range(iterations), disable=not show_progress)

        for it in pbar:

            # Sample a random batch
            idxs = torch.randint(low=0, high=max_start + 1, size=(batch_size,), 
                                 device=dataset.device, dtype=torch.long)
            
            batch = dataset.gather(idxs, window_len)
            
            # Optimization step
            optimizer.zero_grad()

            if variational_loss:
                losses = self.step_elbo(batch)
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
    
