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
from PIL import Image
import argparse
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


def inference(opt,device):
    model_name = opt.model_name
    model = EfficientNet.from_pretrained(model_name, num_classes=opt.num_classes)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()

    model = model.to(device)
    fig = plt.figure()
    running_loss, running_corrects, num_cnt = 0.0, 0, 0

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # path = './ISIC-Archive-Data/00.actinic keratosis/ISIC_0026650.jpeg'
    # path = './ISIC-Archive-Data/11.nevus/ISIC_0000481.jpeg'
    # path = './ISIC-Archive-Data/18.vascular lesion/ISIC_0033762.jpeg'
    path=opt.img_path
    img=Image.open(path)
    inputs=data_transforms(img)
    inputs = inputs.to(device)
    outputs = model(inputs.unsqueeze(0))
    _, preds = torch.max(outputs, 1)
    print(preds)
    # with torch.no_grad():
    #     for i, (inputs, labels) in enumerate(dataloaders[phase]):
    #         print(type(inputs),len(inputs))
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #
    #         outputs = model(inputs)
    #         _, preds = torch.max(outputs, 1)
    #         loss = criterion(outputs, labels)  # batch의 평균 loss 출력
    #
    #         running_loss += loss.item() * inputs.size(0)
    #         running_corrects += torch.sum(preds == labels.data)
    #         num_cnt += inputs.size(0)  # batch size
    #
    #     #         if i == 2: break
    #
    #     test_loss = running_loss / num_cnt
    #     test_acc = running_corrects.double() / num_cnt
    #     print('test done : loss/acc : %.2f / %.1f' % (test_loss, test_acc * 100))
    #
    # # 예시 그림 plot
    # with torch.no_grad():
    #     for i, (inputs, labels) in enumerate(dataloaders[phase]):
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #
    #         outputs = model(inputs)
    #         _, preds = torch.max(outputs, 1)
    #
    #         # 예시 그림 plot
    #         for j in range(1, num_images + 1):
    #             ax = plt.subplot(num_images // 2, 2, j)
    #             ax.axis('off')
    #             ax.set_title('%s : GT %s, Pred %s' % (
    #                 'True' if class_names[str(format(labels[j].cpu().numpy(),'02'))] == class_names[
    #                     str(format(preds[j].cpu().numpy(),'02'))] else 'False',
    #                 class_names[str(format(labels[j].cpu().numpy(),'02'))], class_names[str(format(preds[j].cpu().numpy(),'02'))]))
    #             imshow(inputs.cpu().data[j])
    #         if i == 0: break

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == '__main__':
    opt = argparse.Namespace(
        num_classes=19,
        model_name=str('efficientnet-b0'),
        model_path=str('./classification_model.pt'),
        # img_path=str('./ISIC-Archive-Data/18.vascular lesion/ISIC_0033762.jpeg'),
        # img_path=str('/mnt/ISIC-Archive-Data/00.actinic keratosis/ISIC_0024468.jpeg'),
        img_path=str('/mnt/ISIC-Archive-Data/18.vascular lesion/ISIC_0032867.jpeg'),
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu

    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            inference(opt=opt, device=device)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
