import csv
import os
from imutils import paths
import random

# list all the images in folder ./dataset
img_train = './train_final'
img_name_train = list(paths.list_images(img_train))
random.shuffle(img_name_train)
# print(len(img_name_train))

# divide the dataset into 2 part: training and testing
# Number of training images/ testing iamges ~ 7/3
num_img = len(img_name_train)
num_train = int(num_img*70/100)
# num_train = int(num_img)
print('num_img', num_img)
print('num_train', num_train)
print('num_test', num_img - num_train)


with open('data_train.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(0, num_train):
        img_path = img_name_train[i]
        folder = 'ahead', 'right', 'stop', 'none'
        for i,fd in enumerate(folder):
        	if fd in img_path:
	        	cls_gt = i
	        	writer.writerow([img_path, cls_gt])


with open('data_test.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(num_train, num_img):
        img_path = img_name_train[i]
        folder = 'ahead', 'right', 'stop', 'none'
        for i,fd in enumerate(folder):
	        if fd in img_path:
	            cls_gt = i
	            writer.writerow([img_path, cls_gt])
        
        
# with open('data_test.csv', 'a') as csvfile:
#     writer = csv.writer(csvfile)
#     for i in range(num_train, num_img):

#         img_path = img_name_ck_list[i]
#          # write the testing section
#         if 'noleft' in img_path:
#             cls_gt = 0
#             writer.writerow([img_path, cls_gt])
#         if 'left' in img_path:
#             cls_gt = 1
#             writer.writerow([img_path, cls_gt])
#         if 'ahead' in img_path:
#             cls_gt = 2
#             writer.writerow([img_path, cls_gt])
#         if 'noright' in img_path:
#             cls_gt = 3
#             writer.writerow([img_path, cls_gt])
#         if 'right' in img_path:
#             cls_gt = 4
#             writer.writerow([img_path, cls_gt])
#         if 'stop' in img_path:
#             cls_gt = 5
#             writer.writerow([img_path, cls_gt])
#         if 'none' in img_path:
#             cls_gt = 6
#             writer.writerow([img_path, cls_gt])