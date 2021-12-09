# collate function
# save all images and boxes in the batch in one array
import numpy as np


def ssd_dataset_collate(batch):
    images = []
    boxes = []
    for img, box in batch:
        images.append(img)
        boxes.append(box)
    images = np.array(images)
    boxes = np.array(boxes)
    return images, boxes
