import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Sequence, Union, Optional



# --------------------------------------------------------------------------- #
#                               C Elegans dataset                             #
# --------------------------------------------------------------------------- #
class CElegansDataset(Dataset):
    """
    Concatenate one or more C Elegans recordings and serve them
    either as sliding windows or as a single full sequence.

    Parameters
    ----------
    paths              - directory or list of directories that each contain
                          either *input.csv* (PCA) or *original_data.csv*
    use_pca            - True  → read *input.csv*,
                         False → read *original_data.csv*
    mode               - "window" | "full"
    window_length      - length of each sample (required if mode="window")
    window_shift       - stride between window starts
    standardize        - z-score each channel
    finite_difference  - replace x[t] with x[t]-x[t-1] before std
    downsample_step    - keep every *n*th sample (>1 undersamples)
    device             - move the tensor once to this device
    dtype              - stored dtype
    """

    def __init__(
        self,
        paths: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        use_pca: bool = True,
        mode: str = "window",
        window_length: Optional[int] = None,
        window_shift: int = 1,
        standardize: bool = False,
        finite_difference: bool = False,
        downsample_step: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if mode not in {"window", "full"}:
            raise ValueError('mode must be "window" or "full"')

        file_name = "input.csv" if use_pca else "original_data.csv"
        if isinstance(paths, (str, Path)):
            paths = [paths]

        # ------------------------------ load ------------------------------ #
        series_list = []
        for p in paths:
            p = Path(p)
            f = p / file_name
            if not f.is_file():
                raise FileNotFoundError(f"Missing file {f}")
            arr = np.genfromtxt(f, delimiter=",", dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"{f} must be 2-D, got {arr.shape}")
            series_list.append(arr)

        data = np.concatenate(series_list, axis=0)

        # ----------------------- preprocessing pipeline ------------------- #
        if downsample_step > 1:
            data = data[:: downsample_step]

        if finite_difference:
            data = np.diff(data, axis=0)

        tensor = torch.as_tensor(data, dtype=dtype)

        if standardize:
            m = tensor.mean(0, keepdim=True)
            s = tensor.std(0, keepdim=True).clamp(min=1e-8)
            tensor = (tensor - m) / s

        if device is not None:
            tensor = tensor.to(device)

        self._series = tensor                                 # (T, F)
        self.mode = mode
        self.window_length = window_length
        self.window_shift = window_shift

        T = len(tensor)
        if mode == "window":
            if window_length is None or window_length > T:
                raise ValueError("invalid window_length")
            self._n = (T - window_length) // window_shift + 1
        else:
            self._n = 1

    # --------------------------------------------------------------------- #
    #  PyTorch Dataset API                                                  #
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self._n:
            raise IndexError
        if self.mode == "full":
            return self._series
        start = idx * self.window_shift
        end = start + self.window_length
        return self._series[start:end]                      # view

    # --------------------------------------------------------------------- #
    #  Convenience getters                                                  #
    # --------------------------------------------------------------------- #
    @property
    def series(self) -> torch.Tensor:
        """Full (T, 3) tensor – useful if you need direct access."""
        return self._series

    @property
    def times(self) -> torch.Tensor:
        """1-D tensor of simulation times (assumes constant dt)."""
        return torch.arange(
            0, len(self._series), device=self._series.device, dtype=self._series.dtype
        )

    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        T, F = self._series.shape
        extra = (
            f"windows={self._n}, win_len={self.window_length}, shift={self.window_shift}"
            if self.mode == "window"
            else "full_sequence"
        )
        return f"{self.__class__.__name__}({extra}, time_steps={T}, features={F})"
