import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from AlexNet.alexnet import AlexNet
from GoogleNet.googlenet import GoogLeNet
from ResNet.resnet import ResNet, resnet34
from VGG.vgg import VGG,vgg
from MobileNet.model_v1 import MobileNetV1
from MobileNet.mobilenetv2 import MobileNetV2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_trans = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
# Mototopo
img_path = "20130412_045655_58144.jpg"
assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
img = Image.open(img_path)

plt.imshow(img)

img = data_trans(img)
img = torch.unsqueeze(img, dim=0)

json_path = "ship_classes.json"
assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

json_file = open(json_path, "r")
class_ind = json.load(json_file)

model = vgg(model_name="vgg16", num_classes=14, init_weights=True)

weight_path = "VGG/vgg16Net.pth"
assert os.path.exists(weight_path), "file: '{}' does not exist.".format(weight_path)

model.load_state_dict(torch.load(weight_path))

model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

print_res = "net: MobileNetV2  class: {}   prob: {:.3}".format(class_ind[str(predict_cla)],
                                             predict[predict_cla].numpy())
plt.title(print_res)
print(print_res)
plt.show()