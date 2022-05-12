import torch
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_each_epoch(model, ssd_loss, loss_history, optimizer, current_epoch, epoch_step_train, epoch_step_val,
                     train_dataloader, val_dataloader, Epoch,
                     cuda):
    total_loss = 0
    val_loss = 0

    model.train()
    print('Start Train')
    with tqdm(total=epoch_step_train, desc=f'Epoch {current_epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_dataloader):
            if iteration >= epoch_step_train:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor)
                targets = torch.from_numpy(targets).type(torch.FloatTensor)
                if cuda:
                    images = images.cuda()
                    targets = targets.cuda()

            out = model(images)

            optimizer.zero_grad()

            loss = ssd_loss.forward(targets, out)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {current_epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_dataloader):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor)
                targets = torch.from_numpy(targets).type(torch.FloatTensor)
                if cuda:
                    images = images.cuda()
                    targets = targets.cuda()

                out = model(images)
                optimizer.zero_grad()
                loss = ssd_loss.forward(targets, out)
                val_loss += loss.item()

                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    print('Finish Validation')
    loss_history.append_loss(total_loss / epoch_step_train, val_loss / epoch_step_val)
    print('Epoch:' + str(current_epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step_train, val_loss / epoch_step_val))
    if current_epoch % 50 == 0:
        torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (
            current_epoch + 1, total_loss / epoch_step_train, val_loss / epoch_step_val))
