
import torch
import torch.nn as nn

from torch_points3d.modules.PointTransformer.layers import PointTransformerLayer

N = 5
F_main = 16
F_mod = 7
nc_qk = 8

csr_idx = torch.LongTensor([0, 4, 4, 5, 10, 20])
x_main = torch.rand(N, F_main)
x_mod = torch.rand(N, F_mod)

xyz = torch.rand(N,3)

# This is the offset tensor, it's supposed to contain the count of the
# points that end a batch. So [3, 5] here means points 0,1,2 belong to the
# first batch (count is 3) and 3, 4 belongs to the second one (we count 2 more,
# so it's 5 in total).
o = torch.Tensor([3, 5]).int() 

print(xyz)


pt_layer = PointTransformerLayer(in_planes=F_main, out_planes=16)

pt_layer = pt_layer.cuda()

pt_layer([xyz.cuda(), x_main.cuda(), o.cuda()]) # Need to pass list



print("done!")
 