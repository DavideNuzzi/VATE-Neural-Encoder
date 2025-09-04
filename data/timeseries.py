import torch
import numpy as np
from typing import List
from torch import Tensor
from numpy import ndarray

#TODO: add automatic handling of sting timeseries (convert them into indices, keep a dictionary for conversion)
class Timeseries:
    def __init__(self, 
                 data: Tensor | ndarray | List,
                 name: str):
        
        # Check that the data is valid
        if not isinstance(data, (Tensor, ndarray, list)):
            raise TypeError(f'{name} must be Tensor, ndarray, or list')
        
        # Convert to numpy if list
        if isinstance(data, list):
            data = np.array(data)

        # Handle numpy data and convert to torch
        if isinstance(data, ndarray):
            data = torch.tensor(data, dtype=torch.float32 if data.dtype.kind == 'f' else torch.long)
        # Handle tensor data
        elif data.dtype not in (torch.float32, torch.float64, torch.int64, torch.int32):
            raise TypeError(f'Unsupported tensor dtype: {data.dtype}')
        
        # Check shape of resulting array
        if data.ndim == 1:
            data = data.unsqueeze(1)
        elif data.ndim != 2:
            raise ValueError(f'{name} must be 1D or 2D, got shape {data.shape}')
        
        # Save info
        self.data = data
        self.name = name
        self.channels = data.shape[1]
        self.is_multivariate = self.channels > 1
        self.dtype = 'continuous' if data.dtype in (torch.float32, torch.float64) else 'discrete'

    def to_(self, device: torch.device):
        self.data = self.data.to(device, non_blocking=True)
        return self
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return (f'{self.name} ({self.dtype}, shape={tuple(self.data.shape)}, '
                f'multivariate={self.is_multivariate})')


