
# photos = list()
# labels = list()
#
#
# img1 = cv2.imread('img_detect/img_source/femidol.jpg', 0)
# img1.shape
# img1 = cv2.resize(img1, size)
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img1.shape
# photos.append(img1)
# labels.append(0)
#
# img2 = cv2.imread("img_detect/img_source/femidol2.jpg", 1)
# img2 = cv2.resize(img2, size)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# photos.append(img2)
# labels.append(1)
#
# names = {0: "f1", 1: "f2"}
# reconizer = cv2.face.LBPHFaceRecognizer_create()
# reconizer.train(photos, np.array(labels))
#
# i = cv2.imread('img_detect/box/0.jpg')
# i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
# label, conf = reconizer.predict(i)
# print(str(conf))
# print(names[str(label)])
import os

import cv2
import numpy as np

source_path = 'img_detect'
size = (640,640)
names = {}
images = []
labels = []
train_path = os.path.join(source_path,'img_source')
list_name = os.listdir(train_path)
for i, name in enumerate(list_name):
    names[i] = name
    image_path = os.path.join(train_path, name)
    print(image_path)

