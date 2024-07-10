import cv2
import cvzone
import numpy as np
import time
import pandas as pd
from picamera2 import Picamera2
from ultralytics import YOLO

img_width = 800
img_height = 600

picam2 = Picamera2()
picam2.preview_configuration.main.size = (img_width,img_height)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()


model=YOLO('best.pt')

img_center_x = img_width // 2 - 60
img_center_y = img_height // 2
target_width1 = 640
target_height1 = 448

# 计算目标图像的区域
img_start_x = img_center_x - target_width1 // 2
img_start_y = img_center_y - target_height1 // 2
img_end_x = img_start_x + target_width1
img_end_y = img_start_y + target_height1


#my_file = open("coco.txt", "r")
#data = my_file.read()
#class_list = data.split("\n")

# 设置起始时间
start_time = time.time()
# 帧计数器
frame_count = 0

count=0
while True:
    im= picam2.capture_array()
    target_image1 = im[img_start_y:img_end_y, img_start_x:img_end_x]
    # 调整图像大小
    #resized_image = cv2.resize(image, (target_width, target_height))
    count += 1
    
    #if count % 3 != 0:
    #    continue
    #im=cv2.flip(im,-1)
    results=model.predict(target_image1)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    
    
    for index,row in px.iterrows():
        print(row)

        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        #c=class_list[d]
        
        cv2.rectangle(target_image1,(x1,y1),(x2,y2),(0,0,255),2)
        #cvzone.putTextRect(im,f'{c}',(x1,y1),1,1)
    
    # 计算帧率
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time
     # 在图像上显示帧率
    cv2.putText(target_image1, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Camera", target_image1)
    if cv2.waitKey(1)==ord('q'):
        break
    
    
cv2.destroyAllWindows()
