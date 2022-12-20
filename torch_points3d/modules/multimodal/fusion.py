from abc import ABC

import math

import torch
import torch.nn as nn
from torch_scatter import segment_csr, scatter_min, scatter_max
from torch_points3d.core.common_modules import MLP

from torch_points3d.utils.adl4cv_utils import csr_to_offset

class BimodalFusion(nn.Module, ABC):
    """Bimodal fusion combines features from different modalities into
    a single tensor.

    The input modalities' feature tensors are expected to have matching
    sizes [N x C_1] and [N x C_2]. For residual fusion, we further
    require C_1 = C_2.

    By convention, the second features are fused into the first, main
    modality. This matters as the output format will match that of the
    main modality
    """

    MODES = ['residual', 'concatenation', 'both', 'modality']

    def __init__(self, mode='residual', **kwargs):
        super(BimodalFusion, self).__init__()
        self.mode = mode
        if self.mode == 'residual':
            self.f = lambda a, b: a + b
        elif self.mode == 'concatenation':
            self.f = lambda a, b: torch.cat((a, b), dim=-1)
        elif self.mode == 'both':
            self.f = lambda a, b: torch.cat((a, a + b), dim=-1)
        elif self.mode == 'modality':
            self.f = lambda a, b: b
        else:
            raise NotImplementedError(
                f"Unknown fusion mode='{mode}'. Please choose among "
                f"supported modes: {self.MODES}.")

    def forward(self, x_main, x_mod, xyz):
        if x_main is None:
            return x_mod
        if x_mod is None:
            return x_main

        # If the x_mod is a sparse tensor, we only keep its features
        x_mod = x_mod if isinstance(x_mod, torch.Tensor) else x_mod.F

        # Update the x_main while respecting its format
        x_main = self.f(x_main, x_mod)

        return x_main

    def extra_repr(self) -> str:
        return f"mode={self.mode}"


class SelfAttentiveBimodalFusion(nn.Module, ABC):
    """
    Concatenates the 2d and 3d features to then apply either local or global
    self-attention.

    Implementation inspired by the existing QKVBimodalCSRPool from the original
    authors.

    TODO: For the global attention mode, it would be sensible to add a positional
    encoding.
    """

    MODES = ['global', 'local']

    def __init__(self, mode='global', in_main=None, in_mod=None, out_main=None,
                 nc_inner=16, nc_qk=8):
        super().__init__()
        self.mode = mode
        self.nc_qk = nc_qk
        # Layers
        self.concat = lambda a, b: torch.cat((a, b), dim=-1)

        # E embeds the concatenated features for self-attention
        self.E = MLP([in_main + in_mod, nc_inner, nc_inner], bias=False)

        # Linear transformations for the Query, Key and Values
        self.W_Q = nn.Linear(nc_inner, nc_qk, bias=False)
        self.W_K = nn.Linear(nc_inner, nc_qk, bias=False)
        self.W_V = nn.Linear(nc_inner, out_main, bias=False)

        # Softmax 
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x_main, x_mod, xyz):
        
        # Combine modalities and embed
        x_main = self.concat(x_main, x_mod) # (N, F_main + F_mod)
        x_main = self.E(x_main) # (N, nc_inner)
        
        if self.mode == 'global':
            Q = self.W_Q(x_main) # (N, nc_qk)
            K = self.W_K(x_main) # (N, nc_qk)
            V = self.W_V(x_main) # (N, out_main)

            x_main = self.softmax( Q @ K.T / math.sqrt(self.nc_qk)) @ V
        else:
            pass

        return x_main

