import SSD_backbone
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# copy from the work ParseNet

class L2normalisation(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2normalisation, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SSD(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(SSD, self).__init__()
        self.VGG16 = SSD_backbone.VGG16_mod(pretrained)
        self.extra = SSD_backbone.SSD_extra()
        self.detectors, self.classifiers = SSD_backbone.detector_and_classifier(self.VGG16, self.extra, num_classes)
        self.L2norm = L2normalisation(512, 20)
        # x should be an image with size (300,300,3) or (512,512,3)
        self.num_classes = num_classes
    def forward(self, x):
        # feature_maps from specific layers
        # localization loss (loc) and the confidence loss (conf)
        feature_maps = []
        Lloc = []
        Lconf = []
        # first location 22, i.e, conv 4-3

        for i in range(23):
            x = self.VGG16[i](x)

        # add to feature_maps
        feature_maps.append(self.L2norm(x))

        # conv7
        for i in range(23, len(self.VGG16)):
            x = self.VGG16[i](x)

        feature_maps.append(x)

        extra_index = [2, 6, 10, 14]
        for index, layer in enumerate(self.extra):
            x = layer(x)
            if index in extra_index:
                feature_maps.append(x)

        # reshape
        #Lloc(after) : [num,height,width,channels]
        for x in feature_maps:
            Lloc.append(self.detectors(x).permute(0, 2, 3, 1).contiguous())
            Lconf.append(self.classifiers(x).permute(0, 2, 3, 1).contiguous())
        # reduce dimension and stack the tensor
        #Lloc : (batch_size,num_boxes*4)
        #Lconf : (batch_size,num_boxes*num_classes)
        Lloc = torch.cat([r.view(r.size(0), -1) for r in Lloc], 1)
        Lconf = torch.cat([r.view(r.size(0), -1) for r in Lconf], 1)

        # Lloc: batch_size,num_boxes,4
        # Lconf: batch_size, num_boxes,num_classes
        result = [Lloc.view(Lloc.size(0),-1,4),Lconf.view(Lconf.size(0),-1,self.num_classes)]
        return result

