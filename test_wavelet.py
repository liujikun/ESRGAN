import torch
from pytorch_wavelets import DWTForward, DWTInverse

a = torch.randn(1, 3, 256, 256).cuda()
dwt = DWTForward(J = 1, wave = 'haar').cuda()
DMT1_yl, DMT1_yh = dwt(a)

print(DMT1_yl.shape)
print(len(DMT1_yh))
print(DMT1_yh[0].shape)

b = torch.cat((DMT1_yh[0][:, :, 0, :, :], DMT1_yh[0][:, :, 1, :, :], DMT1_yh[0][:, :, 2, :, :]), 1)
print(b.shape)
