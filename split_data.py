import os
from shutil import copy, rmtree
import random

def mk_dir(file_path: str):
    if os.path.exists(file_path):
        # 文件夹存在则删除原文件夹重创建
        rmtree(file_path)
    os.makedirs(file_path)

# 复制文件到新路径
def copy_pic(current_path: str, new_path: str):
    copy(current_path, new_path)

def main():
    # 设置随机数种子
    random.seed(0)

    # 10%作为验证集
    split_rate = 0.1

    cwd = os.getcwd()
    path_root = os.path.join(cwd, "sc5-2013-Mar-Apr")
    ship_path = os.path.join(path_root, "sc5")
    assert os.path.exists(ship_path), "path '{}' does not exist".format(ship_path)

    ship_class = [cls for cls in os.listdir(ship_path)
                  if os.path.isdir(os.path.join(ship_path, cls))]

    # 创建训练集
    train_path = os.path.join(path_root, "train")
    mk_dir(train_path)
    for cls in ship_class:
        mk_dir(os.path.join(train_path, cls))

    # 创建测试集文件夹
    test_path = os.path.join(path_root, "test")
    mk_dir(test_path)
    for cls in ship_class:
        mk_dir(os.path.join(test_path, cls))

    for cls in ship_class:
        img_path = os.path.join(ship_path, cls)
        images = os.listdir(img_path)
        count = len(images)

        # 划分数据集
        test_index = random.sample(images, k=int(count*split_rate))
        for index, image in enumerate(images):
            if image in test_index:
                # 复制测试集
                current_path = os.path.join(img_path, image)
                new_path = os.path.join(test_path, cls)
                copy(current_path, new_path)
            else:
                # 复制训练集
                current_path = os.path.join(img_path, image)
                new_path = os.path.join(train_path, cls)
                copy(current_path, new_path)
        print()
    print("done")

if __name__ == '__main__':
    main()