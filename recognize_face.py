import cv2
import os
import numpy as np
import json

size = (640, 640)
def train_face(train_path, save_model):
    dic_name = {}
    imges = []
    labels = []
    # path = train_path
    list_name = os.listdir(train_path)
    # print(list_name)
    for i, name in enumerate(list_name):
        dic_name[i] = name
        print(i)
        print(type(i))
        name_path = os.path.join(train_path, name)
        list_img = os.listdir(name_path)
        for each_img in list_img:
            img_path = os.path.join(name_path, each_img)
            img = cv2.imread(img_path)
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            imges.append(img)
            labels.append(i)

    json_str = json.dumps(dic_name)
    with open('facename.json', 'w') as json_file:
        json_file.write(json_str)
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(imges, np.array(labels))
    recognizer.write(save_model)  # .yml文件
    # re_img = cv2.imread('')
    # label, _ = recognizer.predict(re_img)

# train_face('img_train', 'train_model.yml')
def predict_face(save_model, img_detect_path):
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.read(save_model)
    img = cv2.imread(img_detect_path)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    label, _ = recognizer.predict(img)
    with open('facename.json', 'r') as f:
        name = json.load(f)
    print(name[str(label)])

predict_face('train_model.yml', 'box/2.jpg')
