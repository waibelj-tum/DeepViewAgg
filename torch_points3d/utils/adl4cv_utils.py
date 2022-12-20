import torch
import torch.nn as nn

def get_offset_from_xyz(xyz):
    """
    This function extracts the "offset" Tensor that's used in the PointTransformer
    implementation. This variable is basically a cumilative count of the
    number of points in each sample for the current batch.

    The xyz tensor contains the sample index (or batch index) for each point
    in the current batch on it's last column, this function basically 
    counts them up

    -------
    Parameters
    -------

    :param xyz: Tensor, coordinate tensor from torchsparse3d, has the shape
                (N,4), the fourth column corresponds to the batch index for the
                point that corresponds to that row.

    :return offset: Tensor, a 1D tensor that contains the point count where a
                    a sample of the current batch ends. Based off of the
                    convention used in PointTransformer.
    """

    batch_idx = xyz[:,3]
    counts = torch.bincount(batch_idx.int()) #bincount only works for int tensors
    offset = torch.cumsum(counts, dim=0)

    return offset.int() #Need to return int for cuda implmentation of knn

