import numpy as np
import torch
from torch import nn
from torchvision.ops import nms

class BBoxUtility(object):
    def __init__(self, num_classes):
        self.num_classes    = num_classes

    def ssd_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:

            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def decode_boxes(self, mbox_loc, anchors, variances):

        anchor_width     = anchors[:, 2] - anchors[:, 0]
        anchor_height    = anchors[:, 3] - anchors[:, 1]

        anchor_center_x  = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y  = 0.5 * (anchors[:, 3] + anchors[:, 1])


        decode_bbox_center_x = mbox_loc[:, 0] * anchor_width * variances[0]
        decode_bbox_center_x += anchor_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * anchor_height * variances[0]
        decode_bbox_center_y += anchor_center_y


        decode_bbox_width   = torch.exp(mbox_loc[:, 2] * variances[1])
        decode_bbox_width   *= anchor_width
        decode_bbox_height  = torch.exp(mbox_loc[:, 3] * variances[1])
        decode_bbox_height  *= anchor_height


        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height


        decode_bbox = torch.cat((decode_bbox_xmin[:, None],
                                 decode_bbox_ymin[:, None],
                                 decode_bbox_xmax[:, None],
                                 decode_bbox_ymax[:, None]), dim=-1)
        # 防止超出0与1
        decode_bbox = torch.min(torch.max(decode_bbox, torch.zeros_like(decode_bbox)), torch.ones_like(decode_bbox))
        return decode_bbox

    def decode_box(self, predictions, anchors, image_shape, input_shape, letterbox_image, variances = [0.1, 0.2], nms_iou = 0.3, confidence = 0.5):

        mbox_loc        = predictions[0]

        mbox_conf       = nn.Softmax(-1)(predictions[1])

        results = []

        for i in range(len(mbox_loc)):
            results.append([])

            decode_bbox = self.decode_boxes(mbox_loc[i], anchors, variances)

            for c in range(1, self.num_classes):

                c_confs     = mbox_conf[i, :, c]
                c_confs_m   = c_confs > confidence
                if len(c_confs[c_confs_m]) > 0:

                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        nms_iou
                    )

                    good_boxes  = boxes_to_process[keep]
                    confs       = confs_to_process[keep][:, None]
                    labels      = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones((len(keep), 1))

                    c_pred      = torch.cat((good_boxes, labels, confs), dim=1).cpu().numpy()

                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.ssd_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return results

class anchorBox():
    def __init__(self):
        self.image_size = 300
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.feature_maps_sizes = [38, 19, 10, 5, 3, 1]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    def forward(self):
        boxes = []
        for index, value in enumerate(self.feature_maps_sizes):
            box_widths = []
            box_heights = []
            # another reference https://github.com/amdegroot/ssd.pytorch uses different formula to compute the center.
            # pi = si*p(i+1) + ((Kernel(i+1))-1)/2-padding, thus the steps are indicated [8,16,32,64,100,300]

            box_widths.append(self.min_sizes[index])
            box_heights.append(self.min_sizes[index])

            box_widths.append(self.max_sizes[index])
            box_heights.append(self.max_sizes[index])

            for ar in self.aspect_ratios:
                box_widths.append(self.min_sizes[index] * np.sqrt(ar))
                box_heights.append(self.min_size[index] / np.sqrt(ar))

                box_widths.append(self.min_sizes[index] / np.sqrt(ar))
                box_heights.append(self.min_size[index] * np.sqrt(ar))

            box_widths = 0.5 * np.array(box_widths)
            box_heights = 0.5 * np.array(box_heights)

            step = self.image_size / value
            # steps = [8,16,32,64,100,300]
            # step = step[index]

            lin = np.linspace(0.5 * step, self.image_size - 0.5 * step, value)

            centers_x, centers_y = np.meshgrid(lin)
            centers_x = centers_x.reshape(-1, 1)
            centers_y = centers_y.reshape(-1, 1)

            num_anchors_ = len(self.aspect_ratios)
            anchor_boxes = np.concatenate((centers_x, centers_y), axis=1)
            anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchors_))
            # get the offset
            anchor_boxes[:, ::4] -= box_widths
            anchor_boxes[:, 1::4] -= box_heights
            anchor_boxes[:, 2::4] += box_widths
            anchor_boxes[:, 3::4] += box_heights

            anchor_boxes[:, ::2] /= value
            anchor_boxes[:, 1::2] /= value
            anchor_boxes = anchor_boxes.reshape(-1, 4)
            # clip(normalisation)
            anchor_boxes = np.minimum(np.maximum(anchor_boxes, 0.0), 1.0)
            boxes.append(anchor_boxes)
        prior_boxes = np.concatenate(boxes, axis=0)
        return prior_boxes


def get_prior_boxes(backbone='vgg'):
    if backbone == 'vgg':
        prior_boxes = anchorBox().forward()
    return prior_boxes
