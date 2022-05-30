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
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set gpu
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
    # fig = plt.figure()
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
    return preds.item()

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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

if __name__ == '__main__':
    opt = argparse.Namespace(
        num_classes=19,
        model_name=str('efficientnet-b0'),
        model_path=str('./best_accuracy.pt'),
        # img_path=str('./ISIC-Archive-Data/18.vascular lesion/ISIC_0033762.jpeg'),
        img_path=str('/mnt/ISIC-Archive-Data/00.actinic keratosis/ISIC_0024468.jpeg'),
        dir_path=str('/mnt/ISIC-Archive-Data/test/'),
        # gt=str('')
    )
    # print(opt.dir_path)
    # print(os.listdir(opt.dir_path))
    dirList=os.listdir(opt.dir_path)
    lable_list=[]
    path_list=[]

    for i in dirList:
        tempPath=os.path.join(opt.dir_path,i)
        path=tempPath
        # print(i)
        # print(i.split('.'))
        gt=i.split('.')[0]
        # print(gt)
        # print(path)
        tempPath=os.listdir(tempPath)
        # print(tempPath)
        [path_list.append(os.path.join(path,j)) for j in tempPath]
        [lable_list.append(gt) for j in tempPath]
        # [path_list.append(os.path.join(path, j)) for j in tempPath]
        # print(path_list)
        # print(path)
        # print(tempPath)
    print(sorted(set(lable_list)))
    # print(path_list)
    # print(len(path_list))
    # for i, data in enumerate(path_list):
    #     print(data,lable_list[i])
    #
    preds = []
    for i, data in enumerate(path_list):
        opt.img_path = data
        # print(opt.img_path)
        # opt.gt=lable_list[i]
        print(opt.img_path, lable_list[i])
        preds.append(inference(opt=opt, device=device))
        # if(i>5):
        #     break
    # with profile(activities=[
    #     ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         for i, data in enumerate(path_list):
    #             opt.img_path=data
    #             # print(opt.img_path)
    #             # opt.gt=lable_list[i]
    #             print(opt.img_path,lable_list[i])
    #             preds.append(inference(opt=opt, device=device))
    # print(preds)
    pred=[]
    [pred.append(str(format(preds[i],'02'))) for i in range(len(preds))]
    print(pred)
    print(lable_list)
    # lable_list=lable_list[:7]
    print(confusion_matrix(lable_list, pred))
    print("accuracy", accuracy_score(lable_list, pred))
    print("precision", precision_score(lable_list, pred
                                       , average='macro'))
    print("recall", recall_score(lable_list, pred
                                 , average='macro'))
    print("f1_score", f1_score(lable_list, pred
                               , average='macro'))
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
