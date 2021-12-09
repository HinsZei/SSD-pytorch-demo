import torch
import torch.nn as nn
import torch.nn.functional as F


class SSDLoss(nn.Module):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        # set as the paper said, they assume that neg boxes:pos boxes = 3:1,
        # the best alpha is 1.0 through the cross validation.
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception('Please set the class id of background as 0')
        self.background_label_id = background_label_id
        self.negatives_for_hard = torch.FloatTensor([negatives_for_hard])[0]

    def _l1_smooth_loss(self, y_true, y_pred):
        abs_loss = torch.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = torch.where(abs_loss < 1.0, sq_loss, abs_loss - 0.5)
        return torch.sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, min=1e-7)
        softmax_loss = -torch.sum(y_true * torch.log(y_pred),
                                  axis=-1)
        return softmax_loss

    def forward(self, y_true, y_pred):
        #   y_true batch_size, 8732, 4 + self.num_classes + 1 which contains the matching result(y_true[:,:,-1])
        #   y_pred batch_size, 8732, 4 + self.num_classes
        num_boxes = y_true.size()[1]  # 8732
        y_pred = torch.cat([y_pred[0], nn.Softmax(-1)(y_pred[1])], dim=-1)

        #   Smooth L1 Loss, which is proposed in the paper Fast-RCNN, I implemented myself for learning
        #   actually torch offered it as well
        #   batch_size,8732,4 -> batch_size,8732
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                        y_pred[:, :, :4])
        # loc_loss = F.smooth_l1_loss(y_true[:, :, :4], y_pred[:, :, :4])

        #   Cross entropy
        #   batch_size,8732,21 -> batch_size,8732

        conf_loss = self._softmax_loss(y_true[:, :, 4:-1], y_pred[:, :, 4:])

        # compute all the loc loss then pick the loss of pos boxes

        pos_loc_loss = torch.sum(loc_loss * y_true[:, :, -1],
                                 axis=1)
        pos_conf_loss = torch.sum(conf_loss * y_true[:, :, -1],
                                  axis=1)

        #   num_pos     [batch_size,]
        num_pos = torch.sum(y_true[:, :, -1], axis=-1)

        #   num_neg     [batch_size,]
        #  the number of neg boxes must be 3 times of pos boxes, at least, that is a really hard neg training XD
        num_neg = torch.min(self.neg_pos_ratio * num_pos, num_boxes - num_pos)

        pos_num_neg_mask = num_neg > 0

        # if there is no pos boxes then randomly pick 100 boxes as pos

        has_min = torch.sum(pos_num_neg_mask)

        # get the sum of neg boxes of the batch

        num_neg_batch = torch.sum(num_neg) if has_min > 0 else self.negatives_for_hard

        '''If the a priori box/anchor does not contain an object, then the prediction probability that it does not belong to 
        the background is too high, and the sample is hard to classify. so it would be more close a neg sample'''
        index_start = 4 + self.background_label_id + 1
        index_end = index_start + self.num_classes - 1

        #   batch_size,8732
        #   sum up the possibility that the box is not background, larger value means neg
        max_confs = torch.sum(y_pred[:, :, index_start:index_end], dim=2)

        #   # filter out all pos boxes, then do the hard negative mining, i.e, pick num_neg_batch boxes as neg boxes
        max_confs = (max_confs * (1 - y_true[:, :, -1])).view([-1])

        _, indices = torch.topk(max_confs, k=int(num_neg_batch.cpu().numpy().tolist()))

        neg_conf_loss = torch.gather(conf_loss.view([-1]), 0, indices)

        num_pos = torch.where(num_pos != 0, num_pos, torch.ones_like(num_pos))  # N, the number of pos boxes
        total_loss = torch.sum(pos_conf_loss) + torch.sum(neg_conf_loss) + torch.sum(self.alpha * pos_loc_loss)
        # L(x,c,l,g) = (Lconf(x,c) + Lloc(x,l,g))/N
        total_loss = total_loss / torch.sum(num_pos)
        return total_loss