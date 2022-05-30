## 학습 코드
import numpy as np
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from torchvision import transforms, datasets
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.distributed import DistributedSampler
import argparse
import torch.backends.cudnn as cudnn
# from torchvision import transforms, datasets
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated



def train(opt):
    model_name = opt.model_name
    image_size = EfficientNet.get_image_size(model_name)
    model = EfficientNet.from_pretrained(model_name, num_classes=opt.num_classes)
    batch_size = opt.batch_size

    # data_path = './ISIC-Archive-Data'  # class 별 폴더로 나누어진걸 확 가져와서 라벨도 달아준다
    # data_path = '/mnt/ISIC-Archive-Data/'
    data_path = opt.data_path
    print(data_path)

    isic_dataset = torchvision.datasets.ImageFolder(
        data_path,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    ## data split

    train_idx, valid_idx = train_test_split(list(range(len(isic_dataset))), test_size=0.2, random_state=opt.manualSeed)
    print("len(all),len(train_idx),len(tmp_idx): ",len(isic_dataset),len(train_idx),len(valid_idx))
    datasets = {}
    # print(train_idx)
    # print(valid_idx)

    datasets['train'] = Subset(isic_dataset, train_idx)
    datasets['valid'] = Subset(isic_dataset, valid_idx)

    dataloaders, batch_num = {}, {}
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True,
                                                       pin_memory=True, num_workers=opt.workers)
    dataloaders['valid'] = torch.utils.data.DataLoader(datasets['valid'], batch_size=batch_size, shuffle=False,
                                                       pin_memory=True, num_workers=opt.workers)
    batch_num['train'], batch_num['valid'], = len(dataloaders['train']), len(dataloaders['valid'])
    print('batch_size : %d,  num_of_train_batch : %d, num_of_valid_batch : %d' % (batch_size, batch_num['train'], batch_num['valid']))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(),
                             # lr=0.05,
                             lr=opt.lr,
                             momentum=opt.momentum,
                             weight_decay=opt.weight_decay)
    lmbda = lambda epoch: 0.98739
    exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)

    # model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(model, criterion,
    #                                                                                       optimizer_ft,
    #                                                                                       exp_lr_scheduler,
    #                                                                                       dataloaders=dataloaders,
    #                                                                                       device=device,
    #                                                                                       num_epochs=opt.num_iter)
    #
    #학습 파트
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    num_epochs=opt.num_iter
    optimizer=optimizer_ft
    scheduler=exp_lr_scheduler
    from tqdm import tqdm
    for epoch in tqdm(range(num_epochs)):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
            if phase == 'train':
                scheduler.step()

            epoch_loss = float(running_loss / num_cnt)
            epoch_acc = float((running_corrects.double() / num_cnt).cpu() * 100)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print('\n{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc >= best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                #                 best_model_wts = copy.deepcopy(model.module.state_dict())
                print('==> best model saved - %d / %.1f\n' % (best_idx, best_acc))
                torch.save(
                    model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy_{best_acc}.pt')
        # if (epoch + 1) % 1e+5 == 0:
        # if (epoch + 1) % 1 == 0:
        if epoch % 10==0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{epoch+1}.pt')

        if (epoch + 1) == opt.num_iter:
            print('end the training')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' % (best_idx, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'classification_model.pt')
    print('model saved')







    print('best model : %d - %1.f / %.1f' % (best_idx, valid_acc[best_idx], valid_loss[best_idx]))
    fig, ax1 = plt.subplots()

    ax1.plot(train_acc, 'b-', label='train_acc')
    ax1.plot(valid_acc, 'r-', label='valid_acc')
    plt.plot(best_idx, valid_acc[best_idx], 'ro')
    ax1.set_xlabel('epoch')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('acc', color='k')
    ax1.tick_params('y', colors='k')
    ax1.legend(loc='best')
    ax2 = ax1.twinx()
    ax2.plot(train_loss, 'g-', label='train_loss')
    ax2.plot(valid_loss, 'k-', label='valid_loss')
    plt.plot(best_idx, valid_loss[best_idx], 'ro')
    ax2.set_ylabel('loss', color='k')
    ax2.tick_params('y', colors='k')
    ax2.legend(loc='best')
    fig.tight_layout()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    opt = argparse.Namespace(
        num_classes=19,
        class_names={
            # "00": "actinic keratosis",
        "00": "nevus",
        "01": "melanoma",
        "02": "basal cell carcinoma",
        "03": "seborrheic keratosis",
        "04": "pigmented benign keratosis",
        },
        model_name=str('efficientnet-b7'),
        save_model_path=str('./classification_model.pt'),
        data_path=str('/mnt/ISIC-Archive-Data/train_valid3/'),
        # data_path=str('/mnt/ISIC-Archive-Data/train_valid2/'), # 1000개씩
        # data_path=str('./ISIC-Archive-Data'),
        batch_size=int(256),
        manualSeed=int(555),
        workers=int(4),
        num_iter=int(100),
        # valInterval=int(1),
        lr=float(0.01),
        momentum=float(0.9),
        weight_decay=1e-4,
        saved_model=str('./saved_model'),
    )

    print(torch.cuda.get_device_name())
    print("cuda is available ", torch.cuda.is_available())


    opt.exp_name = f'{opt.model_name}'
    opt.exp_name += f'-Seed{opt.manualSeed}'
    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # if opt.num_gpu > 1:
    #     print('------ Use multi-GPU setting ------')
    #     print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
    #     opt.workers = opt.workers * opt.num_gpu
    #     opt.batch_size = opt.batch_size * opt.num_gpu

    train(opt)
