import json
import csv
import numpy as np
import math
import os
import cv2
from collections import OrderedDict


directory = "C:\\Users\\hcaliskan\\Recursive-CNNs-1-new_branch\\highres_double-20210617T092601Z-001\\highres_double\\original_highres_double\\"
# \\GitHub\\DRBox_keras-1\\training_kimlik_new_v1'

# ratios = []

# for filename in os.listdir(directory):
#     if filename.endswith(".jpg"):
#       input_path = os.path.join(directory, filename)
#       img = cv2.imread(input_path, cv2.IMREAD_COLOR)
#       ratio_y = 300/img.shape[0]
#       ratio_x = 300/img.shape[1]
#       img = cv2.resize(img,(300,300))
#       cv2.imwrite((directory+'resized_'+filename), img)
#       ratios.append([ratio_x,ratio_y])
#     else:
#       pass




with open('C:\\Users\\hcaliskan\\Recursive-CNNs-1-new_branch\\highres_double-20210617T092601Z-001\\highres_double\\highres_double\\annotations.json') as f:
  data = json.load(f, object_pairs_hook = OrderedDict)
  

# print(data)
with open("C:\\Users\\hcaliskan\\Recursive-CNNs-1-new_branch\\highres_double-20210617T092601Z-001\\highres_double\\gt.csv", 'w', newline='') as myfile:

  for img_name in data:
      if img_name == '___sa_version___':
          pass
      else:
          input_path = os.path.join(directory, img_name)
          img = cv2.imread(input_path, cv2.IMREAD_COLOR)
          x_image = img.shape[1]
          y_image = img.shape[0]
          ratio_y = 300/img.shape[0]
          ratio_x = 300/img.shape[1]
          img = cv2.resize(img,(300,300))
          cv2.imwrite((directory+'resized_'+img_name), img)
          # print(data[img_name]['instances'])

          sorted_instances = sorted(data[img_name]['instances'], key=lambda s: s['classId'])
          # n=0

          mylist= ['resized_'+img_name+',',
          (np.asarray([sorted_instances[7]['x']*ratio_x,
          sorted_instances[7]['y']*ratio_y]),
          np.asarray([sorted_instances[6]['x']*ratio_x,
          sorted_instances[6]['y']*ratio_y]),
          np.asarray([sorted_instances[5]['x']*ratio_x,
          sorted_instances[5]['y']*ratio_y]),
          np.asarray([sorted_instances[4]['x']*ratio_x,
          sorted_instances[4]['y']*ratio_y]),
          np.asarray([sorted_instances[3]['x']*ratio_x,
          sorted_instances[3]['y']*ratio_y]),
          np.asarray([sorted_instances[2]['x']*ratio_x,
          sorted_instances[2]['y']*ratio_y]),
          np.asarray([sorted_instances[1]['x']*ratio_x,
          sorted_instances[1]['y']*ratio_y]),
          np.asarray([sorted_instances[0]['x']*ratio_x,
          sorted_instances[0]['y']*ratio_y]))]

          mylist_normalized = [img_name+',',
          (np.asarray([sorted_instances[7]['x']/x_image,
          sorted_instances[7]['y']/y_image]),
          np.asarray([sorted_instances[6]['x']/x_image,
          sorted_instances[7]['y']/y_image]),
          np.asarray([sorted_instances[5]['x']/x_image,
          sorted_instances[7]['y']/y_image]),
          np.asarray([sorted_instances[4]['x']/x_image,
          sorted_instances[7]['y']/y_image]),
          np.asarray([sorted_instances[3]['x']/x_image,
          sorted_instances[7]['y']/y_image]),
          np.asarray([sorted_instances[2]['x']/x_image,
          sorted_instances[7]['y']/y_image]),
          np.asarray([sorted_instances[1]['x']/x_image,
          sorted_instances[7]['y']/y_image]),
          np.asarray([sorted_instances[0]['x']/x_image,
          sorted_instances[7]['y']/y_image]))]

          wr = csv.writer(myfile, delimiter='|', quoting=csv.QUOTE_NONE,lineterminator = '|\n')
          wr.writerow(mylist_normalized)

