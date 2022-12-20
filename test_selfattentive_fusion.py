import torch
from torch_points3d.modules.multimodal.fusion import (
    SelfAttentiveBimodalFusion,
    BimodalFusion,
)

N = 12
F_main = 16
F_mod = 7
nc_qk = 8

csr_idx = torch.LongTensor([0, 3, 5])
x_main = torch.rand(N, F_main)
x_mod = torch.rand(N, F_mod)

xyz = torch.rand(N, 3)
batches = torch.Tensor([0,0,0,1,1,1,2,2,3,3,3,3]).view(-1,1)

xyz = torch.cat((xyz, batches), dim=1)

fusion = SelfAttentiveBimodalFusion(
    in_main=F_main, in_mod=F_mod, out_main=F_main, nc_qk=nc_qk, mode='local'
)

fusion = fusion.cuda()

out = fusion(x_main.cuda(), x_mod.cuda(), xyz.cuda())

print('done!')
