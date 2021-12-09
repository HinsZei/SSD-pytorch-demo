import torch.nn as nn


# define modified VGG16 step by step
#   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (1): ReLU(inplace=True)
#   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (3): ReLU(inplace=True) 300*300*64
#   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (6): ReLU(inplace=True)
#   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (8): ReLU(inplace=True) 150*150*128
#   (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): ReLU(inplace=True)
#   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (13): ReLU(inplace=True)
#   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (15): ReLU(inplace=True) 75*75*256
#   (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True) 75+1/2=38
#   (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (18): ReLU(inplace=True)
#   (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (20): ReLU(inplace=True)
#   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (22): ReLU(inplace=True) 38*38*512 for classifier 1
#   (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (25): ReLU(inplace=True)
#   (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (27): ReLU(inplace=True)
#   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (29): ReLU(inplace=True)  19*19*512
#   (30): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False) 19*19*512
#   (31): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
#   (32): ReLU(inplace=True)
#   (33): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
#   (34): ReLU(inplace=True) 19*19*1024 for classifier 2
from torch.hub import load_state_dict_from_url


def VGG16_mod(pretrained=False):
    layer_define = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M_C', 512, 512, 512, 'M',
                    512, 512, 512]
    sequence = []
    in_channel = 3
    # Conv1 - Conv5 in VGG16
    # compare the backbone of 2 methods, SSD set ceil_mode up in the Maxpooling after Conv3
    for layer in layer_define:

        if layer == 'M':
            sequence += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif layer == 'M_C':
            sequence += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            sequence += [nn.Conv2d(in_channel, layer, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_channel = layer

    # use Maxpooling(3,1,1) Convolution layers to replace FC layers

    pooling5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    sequence += [pooling5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    model = nn.ModuleList(sequence)
    # load pretrained weight, idk how to fit the dict in the modified model, so I copied this part from others
    # if pretrained:
    #     state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",
    #                                           model_dir="./model_data")
    #     state_dict = {k.replace('features.', ''): v for k, v in state_dict.items()}
    #     model.load_state_dict(state_dict, strict=False)
    return model


def SSD_extra(in_channels=1024):
    sequence = []
    # for classifier 3 10*10*512
    sequence += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1),nn.ReLU(inplace=True)]
    sequence += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),nn.ReLU(inplace=True)]
    # for classifier 4 5*5*256
    sequence += [nn.Conv2d(512, 128, kernel_size=1, stride=1),nn.ReLU(inplace=True)]
    sequence += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),nn.ReLU(inplace=True)]
    # for classifier 5 3*3*256
    sequence += [nn.Conv2d(256, 128, kernel_size=1, stride=1),nn.ReLU(inplace=True)]
    sequence += [nn.Conv2d(128, 256, kernel_size=3, stride=1),nn.ReLU(inplace=True)]
    # for classifier 6 1*1*256
    sequence += [nn.Conv2d(256, 128, kernel_size=1, stride=1),nn.ReLU(inplace=True)]
    sequence += [nn.Conv2d(128, 256, kernel_size=3, stride=1),nn.ReLU(inplace=True)]

    return nn.ModuleList(sequence)


def detector_and_classifier(VGG_backbone, SSD_extra, num_classes):
    backbone_index = [21, 33]
    extra_index = [2, 6, 10, 14]
    num_boxes = [4, 6, 6, 6, 4, 4]
    detectors = []
    classifiers = []
    for index_box, index in enumerate(backbone_index):  # index : 21,33 num_box: 0,1
        detectors += [nn.Conv2d(VGG_backbone[index].out_channels, num_boxes[index_box] * 4, kernel_size=3,padding= 1)]
        classifiers += [nn.Conv2d(VGG_backbone[index].out_channels,num_boxes[index_box]*num_classes,kernel_size=3,padding=1)]
    for index_box,index in enumerate(extra_index):
        detectors += [nn.Conv2d(SSD_extra[index].out_channels, num_boxes[index_box+2] * 4, kernel_size=3,padding= 1)]
        classifiers += [nn.Conv2d(SSD_extra[index].out_channels,num_boxes[index_box+2]*num_classes,kernel_size=3,padding=1)]

    return nn.ModuleList(detectors),nn.ModuleList(classifiers)

extra_index = [2, 6, 10, 14]
# for index,index_box in enumerate(extra_index):
#     print(index)
#     print(index_box)

backbone_index = [21, 33]
for index_box, index in enumerate(backbone_index):
    print(index)
    print(index_box)