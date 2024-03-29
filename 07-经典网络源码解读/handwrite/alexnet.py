import torch.nn as nn
import torch.utils.model_zoo as model_zoo       #主要用来下载


__all__ = ['Alexnet', 'alexnet']   # from alexnet import *   命令只会import该list内的东西


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class Alexnet(nn.Module):

    def __init__(self, num_classes=1000):
        super(Alexnet, self).__init__()
        self.feautres = nn.Sequential(      # 卷积序列的处理
            nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(64,192, kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classfier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.feautres(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),256*6*6)
        x = self.classfier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    model = Alexnet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

