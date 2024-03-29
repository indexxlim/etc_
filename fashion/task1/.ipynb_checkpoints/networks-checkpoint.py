'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2022, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2022.04.20.
'''
import torch.nn as nn
import torchvision.models as models

from efficientnet_pytorch import EfficientNet
ef_model = EfficientNet.from_name('efficientnet-b0')

class EfNetExtractor(nn.Module):

    def __init__(self):
        super(EfNetExtractor, self).__init__()

        self.efficientnet = EfficientNet.from_name('efficientnet-b0')
        self.modules_front = list(self.efficientnet.children())[:-5]
        self.model_front = nn.Sequential(*self.modules_front)
    def front(self, x):
        """ In the resnet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x)


    
def getModel():
    
    model = EfficientNet.from_pretrained('efficientnet-b0')
    # linear 1408 > 1000
    rel1 = nn.ReLU(inplace=True)
    bn1 = nn.BatchNorm1d(1000)
    drop1 = nn.Dropout(0.25)
    
    lin2 = nn.Linear(1000, 512)
    rel2 = nn.ReLU(inplace=True)
    bn2 = nn.BatchNorm1d(512)
    drop2 = nn.Dropout(0.5)
    
    lin3 = nn.Linear(1280, 7)
    
    return nn.Sequential(model, rel1, bn1, drop1, 
                         lin2, rel2, bn2, drop2, 
                         lin3)    


class ResExtractor(nn.Module):
    """Feature extractor based on ResNet structure
        Selectable from resnet18 to resnet152

    Args:
        resnetnum: Desired resnet version
                    (choices=['18','34','50','101','152'])
        pretrained: 'True' if you want to use the pretrained weights provided by Pytorch,
                    'False' if you want to train from scratch.
    """

    def __init__(self, resnetnum='50', pretrained=True):
        super(ResExtractor, self).__init__()

        if resnetnum == '18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnetnum == '34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnetnum == '50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif resnetnum == '101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif resnetnum == '152':
            self.resnet = models.resnet152(pretrained=pretrained)

        self.modules_front = list(self.resnet.children())[:-2]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        """ In the resnet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x)


class Net_emo(nn.Module):
    """ Classification network of emotion categories based on ResNet18 structure. """
    
    def __init__(self, pre = 'resnet'):
        super(Net_emo, self).__init__()

        if pre=='resnet':
            self.encoder = ResExtractor('18')
            linear_dimension = 512
        elif pre=='efficient':
            self.encoder = EfNetExtractor()
            linear_dimension = 1280
        #self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.daily_linear = nn.Linear(linear_dimension, 7)
        self.gender_linear = nn.Linear(linear_dimension, 6)
        self.embel_linear = nn.Linear(linear_dimension, 3)

    def forward(self, x):
        """ Forward propagation with input 'x' """
        feat = self.encoder.front(x['image'])
        flatten = self.avg_pool(feat).squeeze()

        out_daily = self.daily_linear(flatten)
        out_gender = self.gender_linear(flatten)
        out_embel = self.embel_linear(flatten)

        return out_daily, out_gender, out_embel


if __name__ == '__main__':
    pass

