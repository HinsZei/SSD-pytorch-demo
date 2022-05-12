import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class SSDDataset(Dataset):
    def __init__(self, annotation_lines, image_size, prior_boxes, num_classes, isTrain, overlap_threshold=0.5):
        super(SSDDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)

        self.image_size = image_size
        self.prior_boxes = prior_boxes
        self.num_boxes = len(prior_boxes)
        self.num_classes = num_classes
        self.isTrain = isTrain
        self.overlap_threshold = overlap_threshold

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        image, box = self.get_random_data(self.annotation_lines[index], self.image_size, random=self.isTrain)
        image_data = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        if len(box) != 0:
            boxes = np.array(box[:, :4], dtype=np.float32)

            boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.image_size[1]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.image_size[0]

            one_hot_label = np.eye(self.num_classes - 1)[np.array(box[:, 4], np.int32)]
            box = np.concatenate([boxes, one_hot_label], axis=-1)
        box = self.assign_boxes(box)

        return np.array(image_data), np.array(box)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        line = annotation_line.split()

        image = Image.open(line[0])
        image = cvtColor(image)

        iw, ih = image.size
        h, w = input_shape

        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2


            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)


            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data, box


        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)


        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image


        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)


        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255  # numpy array, 0 to 1


        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def iou(self, box):

        inter_upleft = np.maximum(self.prior_boxes[:, :2], box[:2])
        inter_botright = np.minimum(self.prior_boxes[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]

        area_true = (box[2] - box[0]) * (box[3] - box[1])

        area_gt = (self.prior_boxes[:, 2] - self.prior_boxes[:, 0]) * (self.prior_boxes[:, 3] - self.prior_boxes[:, 1])

        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True, variances=[0.1, 0.1, 0.2, 0.2]):

        iou = self.iou(box)
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))

        assign_mask = iou > self.overlap_threshold


        if not assign_mask.any():
            assign_mask[iou.argmax()] = True


        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]


        assigned_anchors = self.prior_boxes[assign_mask]


        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]

        assigned_anchors_center = (assigned_anchors[:, 0:2] + assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh = (assigned_anchors[:, 2:4] - assigned_anchors[:, 0:2])


        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]
        return encoded_box.ravel()

    def assign_boxes(self, boxes):

        assignment = np.zeros((self.num_anchors, 4 + self.num_classes + 1))
        assignment[:, 4] = 1.0  # suppose all prior anchors are background as initialisation
        if len(boxes) == 0:
            return assignment


        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])

        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)

        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)


        encoded_boxes = encoded_boxes[:, best_iou_mask, :]

        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]

        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]

        assignment[:, -1][best_iou_mask] = 1
        return assignment


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def preprocess_input(inputs):
    MEANS = (104, 117, 123)
    return inputs - MEANS
