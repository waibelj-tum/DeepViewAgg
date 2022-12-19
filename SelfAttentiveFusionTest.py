
import torch
from torch_points3d.modules.multimodal.fusion import SelfAttentiveBimodalFusion, BimodalFusion
from torch_points3d.modules.multimodal.pooling import QKVBimodalCSRPool
from pykeops.torch import LazyTensor
N = 5
F_main = 10
F_mod = 7
F_map = 3
nc_qk = 2

csr_idx = torch.LongTensor([0, 4, 4, 5, 10, 20])
x_main = torch.rand(N, F_main)
x_mod = torch.rand(N, F_mod)
x = torch.rand(V, F_map)

xyz = torch.rand(N,3)
batch_idx = torch.Tensor([0, 0, 1, 1, 2]).view(-1, 1)
xyz = torch.cat((xyz, batch_idx), dim=1)

print(xyz)


fusion = SelfAttentiveBimodalFusion(in_main=F_main, in_mod=F_mod, out_main=10)

fusion(x_main, x_mod, xyz)

print("done!")
 