
import torch
from thop import profile

# Model
from CTCSN import CTCSN

device = torch.device('cuda:0')
print('==> Building model..')
model = CTCSN(snr=0, cr=1, bit_num=10).to(device)

dummy_input = torch.randn(1, 172, 256, 4).to(device)
output, _ = model(dummy_input)
print("模型输出:", output.shape)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))