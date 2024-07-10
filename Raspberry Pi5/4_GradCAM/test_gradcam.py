import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import pytorch_grad_cam
import numpy as np
import cv2

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader
from torchcam.utils import overlay_mask


class myCNN(nn.Module):
  def __init__(self):
    super(myCNN, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
    self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    self.fc_input_size = 16 * 22 * 14
    self.fc1 = nn.Linear(self.fc_input_size, 64) # 8*8*16
    self.fc2 = nn.Linear(64,32)
    self.fc3 = nn.Linear(32,4)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.pool(x)

    x = self.conv2(x)
    x = F.relu(x)
    x = self.pool(x)

    x = x.view(-1, self.fc_input_size)
    x = self.fc1(x)
    x = F.relu(x)

    x = self.fc2(x)
    x = F.relu(x)

    x = self.fc3(x)

    x = F.log_softmax(x, dim=1)

    return x
    
model_path = 'model-20240321.pth'
model = torch.load(model_path)

summary(model, (3, 96, 64))

print(model)

# 遍历模型的每一层，找到最后一个 Conv2d 层
def get_last_conv_layer(model):
    last_conv_layer = None
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer
    return last_conv_layer
    
    
    
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

# 获取最后一个 Conv2d 层
last_conv_layer = get_last_conv_layer(model)
print(last_conv_layer)

img_path = "/home/pi/Python-Workspace/T20240321/dataset_dir/test/Full/change_video3_frame_1390.jpg"
#img_path = "/home/pi/Python-Workspace/T20240321/change_video1_frame_2558.jpg"
img_pil = Image.open(img_path)

input_tensor = test_transform(img_pil).unsqueeze(0)

print("input_tensor: ", input_tensor)
print("input_tensor.shape: ", input_tensor.shape)

model.eval()
# 使用模型进行预测
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    prediction = output.max(1, keepdim=True)[1]
print("output: ",output)
print("probabilities: ", probabilities)
print("prediction: ", prediction)

prediction = prediction.squeeze().item()
print("prediction: ", prediction)
print("Ref: ---> 0: 'Empty', 1: 'Enter', 2: 'Full', 3: 'Leave'}")

target_layers = [model.conv2]
cam = GradCAM(model=model, target_layers=target_layers)
targets = [ClassifierOutputTarget(prediction)]
print(targets)
cam_map = cam(input_tensor=input_tensor, targets=targets)[0]
print("cam_map.shape: ",cam_map.shape)

# 将 PIL 图像转换为 OpenCV 格式（BGR 格式）
img_pil_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
cam_map_cv = cv2.cvtColor(np.array(cam_map), cv2.COLOR_RGB2BGR)

result = overlay_mask(img_pil, Image.fromarray(cam_map), alpha=0.6)
result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
# 在窗口中展示 OpenCV 图像
cv2.imshow('img_pil_cv', img_pil_cv)
cv2.imshow('cam_map_cv', cam_map_cv)
cv2.imshow('result_cv', result_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

result.save('sav_full.jpg')

# 保存 OpenCV 图像为文件
cv2.imwrite('sav_full-cv.jpg', result_cv)
cv2.imwrite('sav_full-orig-cv.jpg', img_pil_cv)
