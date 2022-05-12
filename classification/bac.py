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


def train_model(model, criterion, optimizer, scheduler,dataloaders,device, num_epochs=25):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

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
            print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                #                 best_model_wts = copy.deepcopy(model.module.state_dict())
                print('==> best model saved - %d / %.1f' % (best_idx, best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' % (best_idx, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'classification_model.pt')
    print('model saved')
    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc
def train():
    model_name = 'efficientnet-b0'  # b5

    image_size = EfficientNet.get_image_size(model_name)
    print(image_size)
    model = EfficientNet.from_pretrained(model_name, num_classes=19)
    ## 데이타 로드!!
    batch_size = 256
    random_seed = 555
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    ## make dataset

    # data_path = './ISIC-Archive-Data'  # class 별 폴더로 나누어진걸 확 가져와서 라벨도 달아준다
    data_path = '/mnt/ISIC-Archive-Data/'
    from torchvision import transforms, datasets
    president_dataset = datasets.ImageFolder(
        data_path,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    ## data split

    train_idx, tmp_idx = train_test_split(list(range(len(president_dataset))), test_size=0.2, random_state=random_seed)
    datasets = {}

    datasets['train'] = Subset(president_dataset, train_idx)
    tmp_dataset = Subset(president_dataset, tmp_idx)

    val_idx, test_idx = train_test_split(list(range(len(tmp_dataset))), test_size=0.5, random_state=random_seed)
    datasets['valid'] = Subset(tmp_dataset, val_idx)
    datasets['test'] = Subset(tmp_dataset, test_idx)

    ## data loader 선언
    dataloaders, batch_num = {}, {}
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=8)
    dataloaders['valid'] = torch.utils.data.DataLoader(datasets['valid'], batch_size=batch_size, shuffle=False,pin_memory=True,num_workers=8)
    dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'],batch_size=batch_size, shuffle=False,pin_memory=True,num_workers=8)
    batch_num['train'], batch_num['valid'], batch_num['test'] = len(dataloaders['train']), len(dataloaders['valid']), len(dataloaders['test'])
    print('batch_size : %d,  tvt : %d / %d / %d' % (
    batch_size, batch_num['train'], batch_num['valid'], batch_num['test']))
    num_show_img = 5

    class_names = {

        "00": "actinic keratosis",
        "01": "angiofibroma or fibrous papule",
        "02": "angioma",
        "03": "atypical melanocytic proliferation",
        "04": "basal cell carcinoma",
        "05": "cafe-au-lait macule",
        "06": "dermatofibroma",
        "07": "lentigo NOS",
        "08": "lentigo simplex",
        "09": "lichenoid keratosis",
        "10": "melanoma",
        "11": "nevus",
        "12": "other",
        "13": "pigmented benign keratosis",
        "14": "scar",
        "15": "seborrheic keratosis",
        "16": "solar lentigo",
        "17": "squamous cell carcinoma",
        "18": "vascular lesion",

    }

    # train check
    inputs, classes = next(iter(dataloaders['train']))
    # inp = inputs[0].numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    # plt.imsave('ss.png',inp)
    #
    #
    # print(inputs.shape, classes.shape)
    print(classes)
    # # print(data)
    out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 오려부친다
    imshow(out, title=[class_names[str(format(int(x), '02'))] for x in classes[:num_show_img]])
    # valid check
    inputs, classes = next(iter(dataloaders['valid']))
    out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 오려부친다
    imshow(out, title=[class_names[str(format(int(x), '02'))] for x in classes[:num_show_img]])
    # test check
    inputs, classes = next(iter(dataloaders['test']))
    out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 오려부친다
    imshow(out, title=[class_names[str(format(int(x), '02'))] for x in classes[:num_show_img]])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model.parameters(),
                             # lr=0.05,
                             lr=0.05,
                             momentum=0.9,
                             weight_decay=1e-4)

    lmbda = lambda epoch: 0.98739
    exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)
    print("device: ",device)
    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(model, criterion,
                                                                                          optimizer_ft,
                                                                                          exp_lr_scheduler,
                                                                                          dataloaders=dataloaders,
                                                                                          device=device,
                                                                                          num_epochs=500)
    print('best model : %d - %1.f / %.1f' % (best_idx, valid_acc[best_idx], valid_loss[best_idx]))
    fig, ax1 = plt.subplots()

    ax1.plot(train_acc, 'b-',label='train_acc')
    ax1.plot(valid_acc, 'r-',label='valid_acc')
    plt.plot(best_idx, valid_acc[best_idx], 'ro')
    ax1.set_xlabel('epoch')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('acc', color='k')
    ax1.tick_params('y', colors='k')
    ax1.legend(loc='best')
    ax2 = ax1.twinx()
    ax2.plot(train_loss, 'g-',label='train_loss')
    ax2.plot(valid_loss, 'k-',label='valid_loss')
    plt.plot(best_idx, valid_loss[best_idx], 'ro')
    ax2.set_ylabel('loss', color='k')
    ax2.tick_params('y', colors='k')
    ax2.legend(loc='best')
    fig.tight_layout()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train()
