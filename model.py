import time
import torch
import torch.nn as nn

from torch.nn import init
from utils.res import resnet, hswish, resnetDown
from utils.transformer import transformer
from utils.bridge import resnet2transformer, transformer2resnet
from utils.config import config_294, config_508, config_52

class BaseBlock(nn.Module):
    def __init__(self, inp, exp, out, se, stride, heads, dim):
        super(BaseBlock, self).__init__()
        if stride == 2:
            self.resnet = resnetDown(3, inp, exp, out, se, stride, dim)
        else:
            self.resnet = resnet(3, inp, exp, out, se, stride, dim)
        self.resnet2transformer = resnet2transformer(dim=dim, heads=heads, channel=inp)
        self.transformer = transformer(dim=dim)
        self.transformer2resnet = transformer2resnet(dim=dim, heads=heads, channel=out)

    def forward(self, inputs):
        x, z = inputs
        z_hid = self.resnet2transformer(x, z)
        z_out = self.transformer(z_hid)
        x_hid = self.resnet(x, z_out)
        x_out = self.transformer2resnet(x_hid, z_out)
        return [x_out, z_out]


class resnettransformer(nn.Module):
    def __init__(self, cfg):
        super(resnettransformer, self).__init__()
        self.token = nn.Parameter(nn.Parameter(torch.randn(1, cfg['token'], cfg['embed'])))
       
        self.stem = nn.Sequential(
            nn.Conv2d(3, cfg['stem'], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg['stem']),
            hswish(),
        )
      
        self.bneck = nn.Sequential(
            nn.Conv2d(cfg['stem'], cfg['bneck']['e'], 3, stride=cfg['bneck']['s'], padding=1, groups=cfg['stem']),
            hswish(),
            nn.Conv2d(cfg['bneck']['e'], cfg['bneck']['o'], kernel_size=1, stride=1),
            nn.BatchNorm2d(cfg['bneck']['o'])
        )

     
        self.block = nn.ModuleList()
        for kwargs in cfg['body']:
            self.block.append(BaseBlock(**kwargs, dim=cfg['embed']))
        inp = cfg['body'][-1]['out']
        exp = cfg['body'][-1]['exp']
        self.conv = nn.Conv2d(inp, exp, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(exp)
        self.avg = nn.AvgPool2d((7, 7))
        self.head = nn.Sequential(
            nn.Linear(exp + cfg['embed'], cfg['fc1']),
            hswish(),
            nn.Linear(cfg['fc1'], cfg['fc2'])
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, _, _, _ = x.shape
        z = self.token.repeat(b, 1, 1)
        x = self.bneck(self.stem(x))
        for m in self.block:
            x, z = m([x, z])
     
        x = self.avg(self.bn(self.conv(x))).view(b, -1)
        z = z[:, 0, :].view(b, -1)
        out = torch.cat((x, z), -1)
        return self.head(out)
     


if __name__ == "__main__":
    model = resnettransformer(config_52)
    inputs = torch.randn((3, 3, 224, 224))
    print(inputs.shape)
   
    print("Total number of parameters in networks is {} M".format(sum(x.numel() for x in model.parameters()) / 1e6))
    output = model(inputs)
    print(output.shape)
