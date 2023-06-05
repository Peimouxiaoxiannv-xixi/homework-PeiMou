import torch
import torch.nn as nn
import torch.optim as optim
from tcwnet import TcwNet

from tqdm import *
from torchvision import transforms, datasets, utils
import os
import json
import time
from torch.utils.tensorboard import SummaryWriter

def main():
    batch_size = 4
    lr = 0.0002
    epochs = 60
    save_path = './TcwNet.pth'
    best_acc = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_trans = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
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

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0)
    dataset_test = datasets.ImageFolder(root=img_path + "/test",
                                        transform=data_trans["val"])
    val_times = len(dataset_test)
    val_loader = torch.utils.data.DataLoader(dataset_test,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0)
    print("use {} images for training, {} images for test.".format(train_times, val_times))

    tcw = TcwNet(num_classes=14)

    tcw.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(tcw.parameters(), lr=lr)
    train_steps = len(train_loader)
    writer = SummaryWriter("../TcwNet_logs")
    for epoch in range(epochs):
        tcw.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            img, labels = data
            optimizer.zero_grad()
            output = tcw(img.to(device))
            loss = loss_function(output, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        tcw.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_imgs, val_labels = val_data
                output = tcw(val_imgs.to(device))
                predict = torch.max(output, dim=1)[1]
                acc += torch.eq(predict, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_times
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        writer.add_scalar("TcwNet_loss", running_loss, epoch)
        writer.add_scalar("TcwNet_accuracy", val_accurate, epoch)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(tcw.state_dict(), save_path)
    writer.close()
    print("ResNet34 Best_acc: {}".format(best_acc))
    print('Finished Training')


if __name__ == '__main__':
    main()