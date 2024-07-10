import cv2
from picamera2 import Picamera2
#import pandas as pd
#from ultralytics import YOLO
import cvzone
import numpy as np
import uuid

picam2 = Picamera2()
img_width = 800
img_height = 600
picam2.preview_configuration.main.size = (img_width,img_height)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
#model=YOLO('yolov8n.pt')
#my_file = open("coco.txt", "r")
#data = my_file.read()
#class_list = data.split("\n")
count=0
count_cap=0
img_center_x = img_width // 2 - 50
img_center_y = img_height // 2
target_width1 = 640
target_height1 = 448

target_width2 = 416
target_height2 = 416

target_width3 = 256
target_height3 = 256

# 计算目标图像的区域
img_start_x = img_center_x - target_width1 // 2
img_start_y = img_center_y - target_height1 // 2
img_end_x = img_start_x + target_width1
img_end_y = img_start_y + target_height1

# 计算目标图像的区域
img_start_x2 = img_center_x - target_width2 // 2
img_start_y2 = img_center_y - target_height2 // 2
img_end_x2 = img_start_x2 + target_width2
img_end_y2 = img_start_y2 + target_height2

# 计算目标图像的区域
img_start_x3 = img_center_x - target_width3 // 2
img_start_y3 = img_center_y - target_height3 // 2
img_end_x3 = img_start_x3 + target_width3
img_end_y3 = img_start_y3 + target_height3

# 设置视频编解码器为 MP4V，文件名以及帧率
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 创建三个视频写入对象，每个都具有不同的帧大小
out1 = cv2.VideoWriter('output_256_256.mp4', fourcc, 20.0, (256, 256))
out2 = cv2.VideoWriter('output_416_416.mp4', fourcc, 20.0, (416, 416))
out3 = cv2.VideoWriter('output_640_448.mp4', fourcc, 20.0, (640, 448))
flag = 0
while True:
    im= picam2.capture_array()
    # 提取目标图像区域
    target_image1 = im[img_start_y:img_end_y, img_start_x:img_end_x]
    target_image2 = im[img_start_y2:img_end_y2, img_start_x2:img_end_x2]
    target_image3 = im[img_start_y3:img_end_y3, img_start_x3:img_end_x3]
    count += 1
    
    if flag == 1:
        out1.write(target_image3)
        out2.write(target_image2)
        out3.write(target_image1)
    
    #if count % 3 != 0:
    #    continue
    #im=cv2.flip(im,-1)
    #results=model.predict(im)
    #a=results[0].boxes.data
    #px=pd.DataFrame(a).astype("float")
    
    
    #for index,row in px.iterrows():
#        print(row)
 
        # x1=int(row[0])
        # y1=int(row[1])
        # x2=int(row[2])
        # y2=int(row[3])
        # d=int(row[5])
        # c=class_list[d]
        
        # cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),2)
        # cvzone.putTextRect(im,f'{c}',(x1,y1),1,1)
    cv2.imshow("Camera", target_image1)
    if cv2.waitKey(1)==ord('q'):
        flag = 0
        break
    if cv2.waitKey(1)==ord('s'):
        print("Start Video Save.")
        flag = 1
        # count_cap=count_cap+1
        # # 生成随机文件名
        # random_filename = str(uuid.uuid4()) + '.jpg'
        # # 保存提取的图像
        # cv2.imwrite("cap"+str(count_cap)+"_"+random_filename, target_image1)
        # cv2.imwrite("cap"+str(count_cap)+"__"+random_filename, target_image2)
        # cv2.imwrite("cap"+str(count_cap)+"___"+random_filename, target_image3)
        # #break
        
out1.release()
out2.release()
out3.release()
print("Release All.")
cv2.destroyAllWindows()
