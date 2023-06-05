import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

df_alex = pd.read_csv('NetAcc/run-AlexNet-tag-Alex_accuracy.csv')
df_vgg = pd.read_csv('NetAcc/run-VGG-tag-Vgg_accuracy.csv')
df_google = pd.read_csv('NetAcc/run-GoogleNet-tag-GoogleNet_accuracy.csv')
df_resnet = pd.read_csv('NetAcc/run-ResNet-tag-ResNet_accuracy.csv')
df_mobv1 = pd.read_csv('NetAcc/run-mobilenetv1-tag-MobileNetv1_accuracy.csv')
df_mobv2 = pd.read_csv('NetAcc/run-mobilenetv2-tag-MobileNetv2_accuracy.csv')

step_alex = df_alex['Step'].values.tolist()
acc_alex = df_alex['Value'].values.tolist()

step_vgg = df_vgg['Step'].values.tolist()
acc_vgg = df_vgg['Value'].values.tolist()

step_google = df_google['Step'].values.tolist()
acc_google = df_google['Value'].values.tolist()

step_resnet = df_resnet['Step'].values.tolist()
acc_resnet = df_resnet['Value'].values.tolist()

step_mobv1 = df_mobv1['Step'].values.tolist()
acc_mobv1 = df_mobv1['Value'].values.tolist()

step_mobv2 = df_mobv2['Step'].values.tolist()
acc_mobv2 = df_mobv2['Value'].values.tolist()

plt.xlabel('训练轮数')
plt.ylabel('分类准确度')
plt.title('网络分类准确度对比',fontsize=24)

plt.plot(step_alex, acc_alex, label="AlexNet")
plt.plot(step_vgg, acc_vgg, label="VGG")
plt.plot(step_google, acc_google, label="GoogleNet")
plt.plot(step_resnet, acc_resnet, label="ResNet")
plt.plot(step_mobv1, acc_mobv1, label="MobileNetV1")
plt.plot(step_mobv2, acc_mobv2, label="MobileNetV2")
plt.legend(fontsize=16)
plt.show()

