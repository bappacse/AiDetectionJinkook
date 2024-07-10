import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import os
import glob
import uuid


# def jpg_to_numpy(jpg_file):
    # img = Image.open(jpg_file)
    # img_array = np.array(img)
    # return img_array
    
def crop_and_save_image(image, output_image_path, x, y, width, height):
    #cropped_image = image[y:y+height, x:x+width]
    half_width = width // 2
    half_height = height // 2
    x1 = x - half_width
    y1 = y - half_height
    x2 = x + half_width
    y2 = y + half_height
    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(output_image_path, cropped_image)
    

def jpg_to_numpy(jpg_file):
    with Image.open(jpg_file) as img:
        img_array = np.array(img)
    return img_array

def get_jpg_files_as_numpy(folder_path):
    jpg_files = []
    jpg_arrays = []
    jpg_files.extend(glob.glob(os.path.join(folder_path, '*.jpg')))
    jpg_files.extend(glob.glob(os.path.join(folder_path, '*.JPG')))
    for jpg_file in jpg_files:
        file_name = os.path.basename(jpg_file)
        print(file_name)
        jpg_arrays.append((file_name, jpg_to_numpy(jpg_file)))
        #jpg_arrays.append(jpg_to_numpy(jpg_file))
    
    return jpg_arrays
    
def calculate_center(x1, y1, x2, y2):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y
        
# 指定文件夹路径
folder_path = '/home/pi/Python-Workspace/T20240320/Dataset_20240319/Leave-Total'
jpg_arrays_with_filenames  = get_jpg_files_as_numpy(folder_path)
prepath = './Leave-Total/'
model=YOLO('best.pt')

# 打印所有 JPG 文件的 NumPy 数组形状
#for idx, jpg_array in enumerate(jpg_arrays):
for filename, jpg_array in jpg_arrays_with_filenames:
	jpg_array = cv2.cvtColor(jpg_array, cv2.COLOR_RGB2BGR)
	#print(f"JPG file {idx + 1} shape:", jpg_array.shape)
	jpg_array_copy_np = np.copy(jpg_array)
	jpg_array_copy_np1 = np.copy(jpg_array)
	results=model.predict(jpg_array)
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
		if d==0:
		    # 计算矩形的重心坐标
		    center_x, center_y = calculate_center(x1, y1, x2, y2)
		    output_image_path = prepath + "change_"  +  filename
		    crop_and_save_image(jpg_array, output_image_path, center_x, center_y, 96-32, 96)
		    
		    cv2.rectangle(jpg_array,(x1,y1),(x2,y2),(0,0,255),2)
		    #radius = 45
		    
		    
		    #cv2.imshow("Camera", jpg_array)
		    #if cv2.waitKey()==ord('q'):
		    #    break
		    
		    # 绘制圆
		    # cv2.circle(jpg_array_copy_np, (center_x, center_y), radius, (0, 255, 0), -1)
		    # cv2.imshow("Camera", jpg_array)
		    # cv2.imshow("Circle", jpg_array_copy_np)
		    
		    # output = np.zeros((2*radius, 2*radius, 3), dtype=np.uint8)
		    # # 创建一个黑色背景的掩码
		    # mask = np.zeros_like(jpg_array_copy_np)
		    # # 在掩码上绘制白色的圆形
		    # cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
		    # # 使用掩码来截取圆形区域
		    # circle_part1 = cv2.bitwise_and(jpg_array_copy_np1, mask)
		    
		    # # 创建一个与圆形区域大小相同的空白图像
		    # circle_part = np.zeros((2*radius, 2*radius, 3), dtype=np.uint8)

		    # # 计算在新图像中粘贴圆形区域的起始坐标
		    # start_x = (circle_part.shape[1] - 2*radius) // 2
		    # start_y = (circle_part.shape[0] - 2*radius) // 2

		    # # 将圆形区域从原始图像中截取并粘贴到新图像中
		    # circle_part[start_y:start_y + 2*radius, start_x:start_x + 2*radius] = circle_part1[center_y - radius:center_y+ radius, center_x - radius:center_x + radius]

		    # # 生成随机文件名
		    # random_filename = str(uuid.uuid4()) + '.png'
		    # # 保存提取的图像
		    # cv2.imwrite("./Leave-Total/Circle"+"_"+random_filename, circle_part)
		    # #cv2.imshow("Output", circle_part)
		    # #cv2.waitKey(0)
    
    


