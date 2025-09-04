import torch
from numpy import ndarray
from torch import Tensor
from data.timeseries import Timeseries
from torch.utils.data import Dataset
from typing import Dict, Optional, List, Union, Any
from data.utils import slice_timeseries, slice_timeseries_dict, get_dataset_size


class NeuralDataset(Dataset):
    """
    Holds full neural and label time-series.
    No internal windowing – use .gather(idxs, window_len) to build a batch.

    Args
    ----
    neural           : (T, C) array-like
    target_labels    : dict[name → (T,) or (T, D)]
    nuisance_labels  : optional dict[name → (T,) or (T, D)]
    auto_device      : torch.device | str | None
        * If CUDA and dataset fits, tensors are moved to that device.
        * If None (default) everything stays on CPU.
    gpu_utilisation  : float ∈ (0,1]
        Fraction of free VRAM that we allow the dataset to occupy.
    """
    # --------------------------------------------------------------------- #
    def __init__(self,
                 neural: Union[Tensor, ndarray, List],
                 target_labels: Dict[str, Union[Tensor, ndarray, List]],
                 nuisance_labels: Optional[Dict[str, Union[Tensor, ndarray, List]]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 gpu_utilisation: float = 0.5):

        # Create Timeseries objects
        self.neural = Timeseries(neural, name='neural')

        if not isinstance(target_labels, dict):
            raise TypeError('target_labels must be a dictionary')
        self.target_labels = {n: Timeseries(d, n) for n, d in target_labels.items()}

        if nuisance_labels is not None:
            if not isinstance(nuisance_labels, dict):
                raise TypeError('nuisance_labels must be a dictionary')
            self.nuisance_labels = {n: Timeseries(d, n) for n, d in nuisance_labels.items()}
        else:
            self.nuisance_labels = None

        # Consistency check, all timeseries must share T
        T = len(self.neural)
        lengths = [T] + [len(ts) for ts in self.target_labels.values()]
        if self.nuisance_labels:
            lengths += [len(ts) for ts in self.nuisance_labels.values()]

        if len(set(lengths)) != 1:
            raise ValueError(f'All timeseries must have same length, got {lengths}')

        self.T = T

        # Select the device if not provided
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # If the device a GPU, check if you can move the whole dataset on it
        move_dataset_to_device = True 

        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA not available but device="cuda"')

            move_dataset_to_device = False

            gpu_free_memory, _ = torch.cuda.mem_get_info(self.device)
            dataset_size = get_dataset_size(self)

            if dataset_size <= gpu_utilisation * gpu_free_memory:
                move_dataset_to_device = True
        
        self.dataset_on_device = move_dataset_to_device

        # Move everything on the device
        if move_dataset_to_device:
            self.neural.to_(self.device)
            for ts in self.target_labels.values():
                ts.to_(self.device)
            if self.nuisance_labels:
                for ts in self.nuisance_labels.values():
                    ts.to_(self.device)
            self.device = self.device


    @torch.no_grad()
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single-timepoint sample (no windowing).
        The method is mainly for quick inspection/debug.
        """

        if idx < 0 or idx >= self.T:
            raise IndexError(f'Index {idx} out of range (0, {self.T - 1})')

        return {
            "neural": self.neural.data[idx],                            # (C,)
            "targets": {n: ts.data[idx] for n, ts in self.target_labels.items()},
            "nuisances": None if self.nuisance_labels is None else {
                n: ts.data[idx] for n, ts in self.nuisance_labels.items()}
        }


    @torch.no_grad()
    def gather(self,
               idxs: Union[List[int], torch.Tensor],
               window_len: int) -> Dict[str, Any]:
        """
        Build one mini-batch given start indices and window length.

        idxs       : list[int] | 1-D LongTensor (B,)  — start positions
        window_len : int                               — L
        Returns    : dict with keys 'neural', 'targets', 'nuisances'
        """

        # Convert idxs to tensor on same device as data
        if isinstance(idxs, list):
            idxs = torch.as_tensor(idxs, dtype=torch.long, device=self.device)
        else:
            idxs = idxs.to(self.device, non_blocking=True)

        # Create windowed batches for the neural and label timeseries
        neural_batch = slice_timeseries(self.neural, window_len, idxs, self.device)
        target_batch = slice_timeseries_dict(self.target_labels, window_len, idxs, self.device)
    
        nuisance_batch = None
        if self.nuisance_labels:
            nuisance_batch = slice_timeseries_dict(self.nuisance_labels, window_len, idxs, self.device)

        return {'neural': neural_batch,
                'targets': target_batch,
                'nuisances': nuisance_batch}
    

    def __len__(self) -> int:
        return self.T


    def __repr__(self) -> str:
        lines = [
            'NeuralDataset',
            f'  - {"Timepoints":20}: {self.T}',
            f'  - {"Device":20}: {self.device}',
            f'  - {"Dataset on device?":20}: {self.dataset_on_device}',
            f'  - Neural timeseries\n\t{self.neural}',
             '  - Target labels timeseries'
        ]

        for ts in self.target_labels.values():
            lines.append(f'\t{ts}')
        if self.nuisance_labels:
            lines.append('  - Nuisance labels timeseries')
            for ts in self.nuisance_labels.values():
                lines.append(f'\t{ts}')
        else:
            lines.append('  - Nuisance labels timeseries: None')
        return '\n'.join(lines)