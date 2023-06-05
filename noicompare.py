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

df_alex_noi = pd.read_csv('NetAcc/run-.-tag-Alex_accuracy (1).csv')
df_vgg_noi = pd.read_csv('NetAcc/run-.-tag-Vgg_accuracy (1).csv')
df_google_noi = pd.read_csv('NetAcc/run-.-tag-GoogleNet_accuracy (1).csv')
df_resnet_noi = pd.read_csv('NetAcc/run-.-tag-ResNet_accuracy (1).csv')
df_mobv1_noi = pd.read_csv('NetAcc/run-.-tag-MobileNetv1_accuracy (1).csv')
df_mobv2_noi = pd.read_csv('NetAcc/run-.-tag-MobileNetv2_accuracy (1).csv')

step_alex_noi = df_alex_noi['Step'].values.tolist()
acc_alex_noi = df_alex_noi['Value'].values.tolist()

step_vgg_noi = df_vgg_noi['Step'].values.tolist()
acc_vgg_noi = df_vgg_noi['Value'].values.tolist()

step_google_noi = df_google_noi['Step'].values.tolist()
acc_google_noi = df_google_noi['Value'].values.tolist()

step_resnet_noi = df_resnet_noi['Step'].values.tolist()
acc_resnet_noi = df_resnet_noi['Value'].values.tolist()

step_mobv1_noi = df_mobv1_noi['Step'].values.tolist()
acc_mobv1_noi = df_mobv1_noi['Value'].values.tolist()

step_mobv2_noi = df_mobv2_noi['Step'].values.tolist()
acc_mobv2_noi = df_mobv2_noi['Value'].values.tolist()

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
plt.title('分类准确度对比',fontsize=24)

# plt.plot(step_alex, acc_alex, label="AlexNet")
# plt.plot(step_alex_noi, acc_alex_noi, label="增加噪声后AlexNet")

# plt.plot(step_google, acc_google, label="GoogleNet")
# plt.plot(step_google_noi, acc_google_noi, label="增加噪声后GoogleNet")

# plt.plot(step_mobv1, acc_mobv1, label="mobilenetv1")
# plt.plot(step_mobv1_noi, acc_mobv1_noi, label="增加噪声后mobilenetv1")

# plt.plot(step_mobv2, acc_mobv2, label="mobilenetv2")
# plt.plot(step_mobv2_noi, acc_mobv2_noi, label="增加噪声后mobilenetv2")

# plt.plot(step_resnet, acc_resnet, label="resnet")
# plt.plot(step_resnet_noi, acc_resnet_noi, label="增加噪声后resnet")

plt.plot(step_vgg, acc_vgg, label="VGG16")
plt.plot(step_vgg_noi, acc_vgg_noi, label="增加噪声后VGG16")

plt.legend(fontsize=16)
plt.show()