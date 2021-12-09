import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from Utilitys.Get_Classes import get_classes
from Network.SSD import SSD
from Network.Loss import SSDLoss
from Utilitys.Weight_initial import weights_init
from Utilitys.Box import get_prior_boxes
from Network.LossHistory import LossHistory
from Utilitys.Dataset import SSDDataset
from Utilitys.Dataloader import ssd_dataset_collate
from Utilitys.Train_Epoch import train_each_epoch

if __name__ == "__main__":
    useCuda = torch.cuda.is_available()
    classpath = ''
    pre_trained = False
    useFreezeTraining = True
    weightpath = ''
    image_size = [300, 300]
    # some hyperparams

    Init_Epoch = 1
    Freeze_Epoch = 51
    Freeze_batch_size = 16
    Freeze_lr = 5e-4

    Unfreeze_Epoch = 101
    Unfreeze_batch_size = 8
    Unfreeze_lr = 1e-4

    num_CPU = 8  # my chip is AMD 3900X, suit yourself

    trainingset_annotation_path = ''
    valset_annotation_path = ''

    class_names, num_classes = get_classes(classes_path=classpath)
    num_classes += 1
    prior_boxes = get_prior_boxes()

    model = SSD(num_classes, pre_trained)
    if not pre_trained:
        weights_init(model)
    if weightpath != '':
        print('load pretrained weight')
        device = torch.device('cuda' if useCuda else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weightpath, map_location=device)
        # load weight from conv1-1 to 4-2
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    # start training
    model = model.train()

    if useCuda:
        model = model.cuda()
        cudnn.benchmark = True

    criterion = SSDLoss(num_classes)
    lossHistory = LossHistory('logs/')

    with open(trainingset_annotation_path) as f:
        train_lines = f.readlines()
    with open(valset_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # current training strategy: freeze

    if useFreezeTraining:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch
    else:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Init_Epoch
        end_epoch = Unfreeze_Epoch

        epoch_setp_train = num_train // batch_size
        epoch_setp_val = num_val // batch_size
        if epoch_setp_train == 0 or epoch_setp_val == 0:
            raise ValueError('the dataset is too small')
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
        train_dataset = SSDDataset(train_lines, image_size, prior_boxes, num_classes, True)
        val_dataset = SSDDataset(val_lines, image_size, prior_boxes, num_classes, False)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_CPU,
                                      drop_last=True, collate_fn=ssd_dataset_collate(), pin_memory=True)
        val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_CPU,
                                    drop_last=True, collate_fn=ssd_dataset_collate(), pin_memory=True)
        if useFreezeTraining:
            for param in model.VGG16[:28].parameters():

                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            train_each_epoch(model, criterion, lossHistory, optimizer, epoch, epoch_setp_train, epoch_setp_val,
                             train_dataloader, val_dataloader, end_epoch, useCuda)
            lr_scheduler.step()

        if useFreezeTraining:
            batch_size = Unfreeze_batch_size
            lr = Unfreeze_lr
            start_epoch = Freeze_Epoch
            end_epoch = Unfreeze_Epoch

            epoch_setp_train = num_train // batch_size
            epoch_setp_val = num_val // batch_size
            if epoch_setp_train == 0 or epoch_setp_val == 0:
                raise ValueError('the dataset is too small')
            optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_CPU,
                                          drop_last=True, collate_fn=ssd_dataset_collate(), pin_memory=True)
            val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_CPU,
                                        drop_last=True, collate_fn=ssd_dataset_collate(), pin_memory=True)

            for param in model.VGG16[:28].parameters():
                param.requires_grad = True

            for epoch in range(start_epoch, end_epoch):
                train_each_epoch(model, criterion, lossHistory, optimizer, epoch, epoch_setp_train, epoch_setp_val,
                                 train_dataloader, val_dataloader, end_epoch, useCuda)
                lr_scheduler.step()
