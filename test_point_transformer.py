
import torch
import torch.nn as nn
from torch_points3d.modules.multimodal.fusion import SelfAttentiveBimodalFusion, BimodalFusion
from torch_points3d.modules.multimodal.pooling import QKVBimodalCSRPool
from pykeops.torch import LazyTensor

from point_transformer.pointops.functions import pointops

from torch_points3d.modules.PointTransformer.layers import PointTransformerLayer

N = 5
F_main = 16
F_mod = 7
nc_qk = 8

csr_idx = torch.LongTensor([0, 4, 4, 5, 10, 20])
x_main = torch.rand(N, F_main)
x_mod = torch.rand(N, F_mod)

xyz = torch.rand(N,3)
batch_idx = torch.Tensor([0, 0, 1, 1, 2]).view(-1, 1).long()


o = torch.Tensor([3,5]).int()
#xyz = torch.cat((xyz, batch_idx), dim=1)

print(xyz)


#fusion = SelfAttentiveBimodalFusion(in_main=F_main, in_mod=F_mod, out_main=10)

#fusion(x_main, x_mod, xyz)

pt_layer = PointTransformerLayer(in_planes=F_main, out_planes=16)

pt_layer = pt_layer.cuda()

pt_layer([xyz.cuda(), x_main.cuda(), o.cuda()])



print("done!")
 