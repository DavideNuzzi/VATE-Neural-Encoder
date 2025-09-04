import torch
from data.timeseries import Timeseries
from typing import Dict, List, Union


def slice_timeseries(timeseries: Timeseries,
                     window_len: int,
                     idxs: Union[List[int], torch.Tensor],
                     device: Union[str, torch.device]):
        
        batch_size = idxs.numel()

        # Initialize the data buffer on the device
        buffer = torch.empty((batch_size, window_len, timeseries.channels),
                            dtype=timeseries.data.dtype,
                            device=device)
        
        # Create the batch out of windows
        for b, s in enumerate(idxs.tolist()):
            buffer[b] = timeseries.data[s:s + window_len]

        return buffer


def slice_timeseries_dict(timeseries_dict: Dict[str, Timeseries],
                          window_len: int, 
                          idxs: Union[List[int], torch.Tensor],
                          device: Union[str, torch.device]):

    return {name: slice_timeseries(timeseries_dict[name], window_len, idxs, device) for name in timeseries_dict}


def get_dataset_size(dataset):
     
    neural_bytes = dataset.neural.data.element_size() * dataset.neural.data.nelement()
    target_labels_bytes = sum(ts.data.element_size() * ts.data.nelement() for ts in dataset.target_labels.values())
    nuisance_labels_bytes = 0
    if dataset.nuisance_labels is not None:
         nuisance_labels_bytes = sum(ts.data.element_size() * ts.data.nelement() for ts in dataset.nuisance_labels.values())
    return neural_bytes + target_labels_bytes + nuisance_labels_bytes
    

def move_to_device(obj, device):
    """
    Recursively move all tensors in `obj` to the specified device.

    Supports:
    - Tensors
    - Lists and tuples
    - Dictionaries (including nested dicts)
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    else:
        return obj