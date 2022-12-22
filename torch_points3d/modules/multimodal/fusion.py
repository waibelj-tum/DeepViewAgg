from abc import ABC

import math

import torch
import torch.nn as nn
from torch_scatter import segment_csr, scatter_min, scatter_max
from torch_points3d.core.common_modules import MLP

from torch_points3d.modules.PointTransformer.layers import PointTransformerLayer
from torch_points3d.utils.adl4cv_utils import get_offset_from_xyz


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

    MODES = ["residual", "concatenation", "both", "modality"]

    def __init__(self, mode="residual", **kwargs):
        super(BimodalFusion, self).__init__()
        self.mode = mode
        if self.mode == "residual":
            self.f = lambda a, b: a + b
        elif self.mode == "concatenation":
            self.f = lambda a, b: torch.cat((a, b), dim=-1)
        elif self.mode == "both":
            self.f = lambda a, b: torch.cat((a, a + b), dim=-1)
        elif self.mode == "modality":
            self.f = lambda a, b: b
        else:
            raise NotImplementedError(
                f"Unknown fusion mode='{mode}'. Please choose among "
                f"supported modes: {self.MODES}."
            )

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

    Implementation and notation inspired by the existing QKVBimodalCSRPool
    from the original authors.

    TODO: For the global attention mode, it would be sensible to add a positional
    encoding.
    
    TODO: Global attention causes CUDA memory overflow, do we even separate the
    samples in the batches? Lowering the dimensions didn't help

    TODO: Local attention also has memory issues even when the batch size is 2,
    batch size 1 seems to work.
    """

    MODES = ["global", "local"]

    def __init__(
        self,
        mode=None,
        in_main=None,
        in_mod=None,
        out_main=None,
        nc_inner=16,
        nc_qk=8,
        nsample=16,
        **kwargs
    ):
        super().__init__()
        self.mode = mode
        self.in_main = in_main
        self.in_mod = in_mod
        self.out_main = out_main
        self.nc_inner = nc_inner
        self.nc_qk = nc_qk
        self.nsample = nsample

        # Layers
        self.concat = lambda a, b: torch.cat((a, b), dim=-1)

        # E embeds the concatenated features for self-attention
        self.E = MLP([in_main + in_mod, nc_inner, nc_inner], bias=False)

        if self.mode == "global":
            # Linear transformations for the Query, Key and Values
            self.W_Q = nn.Linear(nc_inner, nc_qk, bias=False)
            self.W_K = nn.Linear(nc_inner, nc_qk, bias=False)
            self.W_V = nn.Linear(nc_inner, out_main, bias=False)

            # Softmax
            self.softmax = nn.Softmax(dim=1)

        elif self.mode == "local":
            # PointTransformer layer, it already has linear layers and softmax.
            # Works on the embedded representations, so in_planes is nc_inner.
            self.pointtransformer_layer = PointTransformerLayer(
                in_planes=self.nc_inner, out_planes=self.out_main, nsample=self.nsample
            )

        else:
            raise NotImplementedError(f"Unsupported fusion mode {self.mode}!")

    def forward(self, x_main, x_mod, xyz):
        # Since the whole architecture is built of MultiModalDown/Up blocks
        # all of them have a fusion module even if there is no extra modality.
        # This bit of code is necessary to skip the blocks where there is no 
        # branching.
        if x_main is None:
            return x_mod
        if x_mod is None:
            return x_main

        # Combine modalities and embed
        x_fused = self.concat(x_main, x_mod)  # (N, F_main + F_mod)
        x_fused = self.E(x_fused)  # (N, nc_inner)

        if self.mode == "global":
            Q = self.W_Q(x_fused)  # (N, nc_qk)
            K = self.W_K(x_fused)  # (N, nc_qk)
            V = self.W_V(x_fused)  # (N, out_main)

            x_fused = self.softmax(Q @ K.T / math.sqrt(self.nc_qk)) @ V

        elif self.mode == "local":
            # Fourth column is the batch idx we need coordinates
            coords = xyz[:, 0:3]

            # Knn query pointops expects a float tensor for the coords
            coords = coords.float()

            # All tensors must be contiguous otherwise Cuda problems
            coords = coords.contiguous() 

            offset = get_offset_from_xyz(xyz)  # Conversion of notation

            # PointTransformer layer expects a list of coordinates, features and
            # the offset tensor.
            x_fused = self.pointtransformer_layer([coords, x_fused, offset])

        else:
            raise NotImplementedError(f"Unsupported fusion mode {self.mode}!")

        return x_fused
