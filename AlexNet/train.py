import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import *

from alexnet import AlexNet
from torchvision import transforms, datasets, utils
import os
import json
import time
import matplotlib
from torch.utils.tensorboard import SummaryWriter
from addnoise import AddSaltPepperNoise

def main():
    batch_size = 4
    lr = 0.0002
    epochs = 60
    save_path = './AlexNet.pth'
    best_acc=0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    transforms.ToTensor(),

    data_trans = {
        "train": transforms.Compose([
                                    AddSaltPepperNoise(0.2),
                                    transforms.RandomResizedCrop(224),     # 随机裁剪
                                     transforms.RandomHorizontalFlip(),     # 水平方向随机翻转
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
    img_path = os.path.join(data_root, "sc5-2013-Mar-Apr")
    # 载入训练数据
    dataset_train = datasets.ImageFolder(root=img_path + "/train",
                                         transform=data_trans["train"])
    train_times = len(dataset_train)

    ship_list = dataset_train.class_to_idx
    # 获取类别字典，将类别作为键值
    cls_dic = dict((v, k) for k, v in ship_list.items())

    # indent参数决定空格数
    json_txt = json.dumps(cls_dic, indent=4)
    with open('./ship_classes.json', 'w') as json_file:
        json_file.write(json_txt)



    train_data_loader = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0)
    dataset_test = datasets.ImageFolder(root=img_path + "/test",
                                        transform=data_trans["val"])
    test_times = len(dataset_test)
    test_data_loader = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=0)

    print("use {} images for training, {} images for test.".format(train_times, test_times))

    alex = AlexNet(cls=14, init_weights=True)

    alex.to(device)
    loss_f = nn.CrossEntropyLoss()

    optimizer = optim.Adam(alex.parameters(), lr=lr)

    train_loss = []
    train_accuracy = []
    steps = len(train_data_loader)
    writer = SummaryWriter("../AlexNet_logs")
    for epoch in range(epochs):
        # 管理dropout层
        alex.train()
        run_loss = 0.0
        train_progress_bar = tqdm(train_data_loader)
        for step, data in enumerate(train_progress_bar):
            train_img, train_labels = data
            optimizer.zero_grad()
            outputs = alex(train_img.to(device))
            loss = loss_f(outputs, train_labels.to(device))
            loss.backward()
            optimizer.step()

            # 运行时损失
            run_loss += loss.item()

            # 训练进度
            train_progress_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        alex.eval()
        acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_data_loader)
            for test_data in test_bar:
                test_img, test_label = test_data
                outputs = alex(test_img.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, test_label.to(device)).sum().item()

        test_acc = acc / test_times
        print('[epoch %d] train_loss: %.3f test_accuracy: %.3f' %
            (epoch + 1, run_loss / steps, test_acc))
        train_loss.append(run_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(alex.state_dict(), save_path)

        writer.add_scalar("Alex_loss", run_loss, epoch)
        writer.add_scalar("Alex_accuracy", test_acc, epoch)

    writer.close()
    print("AlexNet Best_acc: {}".format(best_acc))
    print('done')



if __name__ == '__main__':
    main()


