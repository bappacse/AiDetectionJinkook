import cv2
import cvzone
import numpy as np
import time
import pandas as pd
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import pytorch_grad_cam
import threading
import uuid

from picamera2 import Picamera2
from ultralytics import YOLO
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
    self.fc_input_size = 12 * 16 * 16
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
    
model_path = '/home/pi/Python-Workspace/T20240416/model-20240328_99pp.pth'
model_cata = torch.load(model_path,map_location=torch.device('cpu'))

summary(model_cata, (3, 72, 54))

print(model_cata)

def predict_model(model, input_tensor):
    """
    使用模型进行推理并返回预测结果。
    
    参数：
        - model: 使用的 PyTorch 模型
        - input_tensor: 输入张量，要求形状为 (batch_size, channels, height, width)
    
    返回值：
        - prediction: 预测结果张量，形状为 (batch_size,)
        - probabilities: 类别概率张量，形状为 (batch_size, num_classes)，其中 num_classes 是模型输出的类别数
    """
    # 将模型设置为评估模式
    model.eval()
    
    # 禁用梯度计算
    with torch.no_grad():
        # 使用模型进行推理
        output = model(input_tensor)
        
        # 计算类别概率
        probabilities = torch.softmax(output, dim=1)
        
        # 获取预测类别
        prediction = output.max(1, keepdim=True)[1].squeeze()
    
       
    
# model.eval()
                
                # with torch.no_grad():
                    # output = model(input_tensor)
                    # probabilities = torch.softmax(output, dim=1)
                    # prediction = output.max(1, keepdim=True)[1]
    # 返回预测结果
    return prediction, probabilities
    

# 遍历模型的每一层，找到最后一个 Conv2d 层
def get_last_conv_layer(model):
    last_conv_layer = None
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer
    return last_conv_layer


def crop_image(image, x, y, width, height):
    bias_y = 6 # down +
    bias_x = 3 # right +
    half_width = width // 2
    half_height = height // 2
    x1 = x - half_width + bias_x
    y1 = y - half_height + bias_y
    x2 = x + half_width + bias_x
    y2 = y + half_height + bias_y

    #print("x1:", crop_x1, "x2:", crop_x2, "y1:", crop_y1, "y2", crop_y2)
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image, x1, y1, x2, y2
    
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

# 获取最后一个 Conv2d 层
last_conv_layer = get_last_conv_layer(model_cata)
print(last_conv_layer)   

img_width = 800
img_height = 600

picam2 = Picamera2()
picam2.preview_configuration.main.size = (img_width,img_height)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()


img_center_x = img_width // 2 - 62
img_center_y = img_height // 2 - 7 # - <--> down, + up
target_width1 = 224
target_height1 = 224

detect_img_width = 54
detect_img_height = 72

# 计算目标图像的区域
img_start_x = img_center_x - target_width1 // 2
img_start_y = img_center_y - target_height1 // 2
img_end_x = img_start_x + target_width1
img_end_y = img_start_y + target_height1

# 颜色定义（BGR格式）
colors = {
    'Green': (0, 255, 0),
    'Orange': (0, 165, 255),
    'Yellow': (0, 255, 255),
    'Blue': (255, 0, 0)
}

# 设置起始时间
start_time = time.time()
# 帧计数器
frame_count = 0
timeout_ms = 200  # 毫秒
case_chamber = ['Empty', 'Empty_N','Enter', 'Full', 'Leave']
case_chamber_color = ['Green', 'Green','Orange', 'Yellow', 'Blue']
count=0
frame_count = 0
gradcam_flag = 1
count_cap=0

crop_x1 = 0
crop_x2 = 0
crop_y1 = 0
crop_y2 = 0

while True:
    im= picam2.capture_array()
    target_image1 = im[img_start_y:img_end_y, img_start_x:img_end_x]
    
    count += 1
    
    if count < 30:
        continue
        
    if gradcam_flag>0:
        image_copy = target_image1.copy()
        center_x = 101
        center_y = 88
        cropped_img, crop_x1, crop_y1, crop_x2, crop_y2 = crop_image(image_copy, center_x, center_y, detect_img_width, detect_img_height)
        pil_cropped_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        input_tensor = test_transform(pil_cropped_img).unsqueeze(0)
        
        model_cata.eval()
        with torch.no_grad():
            output = model_cata(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.max(1, keepdim=True)[1]
        prediction = prediction.squeeze().item()
        target_layers = [model_cata.conv2]
        cam = GradCAM(model=model_cata, target_layers=target_layers)
        targets = [ClassifierOutputTarget(prediction)]
        cam_map = cam(input_tensor=input_tensor, targets=targets)[0]
        img_pil_cv = cv2.cvtColor(np.array(pil_cropped_img), cv2.COLOR_RGB2BGR)
        cam_map_cv = cv2.cvtColor(np.array(cam_map), cv2.COLOR_RGB2BGR)
        result = overlay_mask(pil_cropped_img, Image.fromarray(cam_map), alpha=0.6)
        result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        gradcam_result = result_cv.copy()
        original_width = detect_img_width
        original_height = detect_img_height
        target_height = 224
        aspect_ratio = original_width / original_height
        target_width = int(target_height * aspect_ratio)
        gradcam_result_resized_img = cv2.resize(gradcam_result, (target_width, target_height))
        
    
    if gradcam_flag>0:
        cv2.rectangle(target_image1,(crop_x1,crop_y1),(crop_x2,crop_y2),colors[case_chamber_color[prediction]],2)
        #print("x1:", crop_x1, "x2:", crop_x2, "y1:", crop_y1, "y2", crop_y2)
        cv2.putText(target_image1, f"Prediction: {case_chamber[prediction]}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[case_chamber_color[prediction]], 2)
        
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time
    cv2.putText(target_image1, f"FPS: {round(fps, 2)}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if gradcam_flag>0:
        total_width = image_copy.shape[1] + target_image1.shape[1] + gradcam_result_resized_img.shape[1]
        merged_img = np.zeros((224, total_width, 3), dtype=np.uint8)
        # 定位并粘贴每个图像
        x_offset = 0
        merged_img[:, x_offset:x_offset + image_copy.shape[1]] = image_copy
        x_offset += image_copy.shape[1]
        merged_img[:, x_offset:x_offset + target_image1.shape[1]] = target_image1
        x_offset += target_image1.shape[1]
        merged_img[:, x_offset:x_offset + gradcam_result_resized_img.shape[1]] = gradcam_result_resized_img
    
    cv2.imshow("Camera", target_image1)
    if gradcam_flag>0:
        #cv2.imshow("GradCAM", gradcam_result_resized_img)
        cv2.imshow("Camera", merged_img)
    if cv2.waitKey(1)==ord('s'):
        count_cap=count_cap+1
        # 生成随机文件名
        random_filename = str(uuid.uuid4()) + '.jpg'
        # 保存提取的图像
        #cv2.imwrite("cap"+str(count_cap)+"_"+random_filename, image_copy)
        cv2.imwrite("crop_cap"+str(count_cap)+"__"+random_filename, cropped_img)
    if cv2.waitKey(1)==ord('q'):
        break
