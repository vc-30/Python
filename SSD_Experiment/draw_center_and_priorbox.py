##############################################################################################################
#    This script is useful to visualize how the default prior boxes and its center would appear.             #
#    Very useful in analyzing scales and featuremap when trying on custom dataset.                           #   
#    The script is following the code from original priorbox layer caffe implementation available at -       #
#    https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/prior_box_layer.cpp                         #
##############################################################################################################

import matplotlib.pyplot as plt 
import cv2
import numpy as np
import math

source_image = cv2.imread("/path/to/original.jpg")

#network resolution
img_height, img_width, _ = source_image.shape

#prior box layer dimensions
# layer_width_VGA = [40,40,20,10,8,6]
# layer_height_VGA = [30,30,15,8,6,4]

#Different feature maps for 1MP input image
layer_width_1MP = [80,80,40,20,18,16]   
layer_height_1MP = [45,45,23,12,10,8]

#Min sizes following SSD default values
min_sizes = [30.0, 60.0, 111.0, 162.0, 213.0, 264.0] #[21.28, 45.6, 91.2, 136.8, 182.4, 228.0, 273.6]
aspect_ratio = [2, 3, 0.5, 0.33]    #normal and its reciprocal

offset = 0.5
for indx in range(6):
    layer_width = layer_width_1MP[indx] 
    layer_height = layer_height_1MP[indx]
    step_w = float(img_width) / layer_width
    step_h = float(img_height) / layer_height
    X, Y = [], []
    for h in range(layer_height):
        for w in range(layer_width):
            center_x = (w + offset) * step_w
            center_y = (h + offset) * step_h 
            # print(center_x, center_y)
            X.append(center_x) 
            Y.append(center_y)     
    # print("======================================")
    # plotting the points  
    plt.scatter(X, Y, s=0.1) 
    plt.xlabel('img_width') 
    plt.ylabel('img_height') 
    plt.title('PriorBox %d centers'%(indx+1)) 
    
    #Uncomment below to visualize only center points
    # plt.show() 

    if indx == 2: #taking higher index to in order to visualize bigger prior box
        ##########################################################################################################################
        
        #box1 with aspect_ratio = 1
        print(len(X))
        p_center_x = 766 #X[int(len(X)/2)]#X[int(len(X)/4)]
        p_center_y = 355 #Y[int(len(Y)/2)]#Y[int(len(Y)/4)]
        print(p_center_x, p_center_y)
        box1_w, box1_h = min_sizes[indx],min_sizes[indx]
        xmin,ymin, xmax, ymax = (p_center_x - box1_w/2), (p_center_y - box1_h/2), (p_center_x + box1_w/2), (p_center_y + box1_h/2)
        print(xmin, ymin, xmax, ymax)
        cv2.rectangle(source_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), 2)
        #box1 ends

        #box2 with aspect_ratio = sqrt(min_size*max_size)
        max_size = min_sizes[indx+1]
        box1_h, box1_w = int(math.sqrt(min_sizes[indx]*max_size)), int(math.sqrt(min_sizes[indx]*max_size))
        xmin,ymin, xmax, ymax = (p_center_x - box1_w/2), (p_center_y - box1_h/2), (p_center_x + box1_w/2), (p_center_y + box1_h/2)
        print(xmin, ymin, xmax, ymax)
        cv2.rectangle(source_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
        #box2 ends

        #box using aspect ratio
        extra = 30
        for ix,ar in enumerate(aspect_ratio):
            box1_w = int(min_sizes[indx] * math.sqrt(ar))
            box1_h = int(min_sizes[indx] * (1 / math.sqrt(ar)))
            xmin,ymin, xmax, ymax = (p_center_x - box1_w/2), (p_center_y - box1_h/2), (p_center_x + box1_w/2), (p_center_y + box1_h/2)
            print(xmin, ymin, xmax, ymax)
            extra += ix*20
            cv2.rectangle(source_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (128+extra,0,0), 2)
        #box using aspect ratio ends

        #########################################################################################################################
        int_X = [int(val) for val in X]
        int_Y = [int(val) for val in Y]
        # for idx,_ in enumerate(int_X):
        #     cv2.circle(source_image, (int_X[idx], int_Y[idx]), 1, (0, 255, 0), -1)
        source_image[int_Y, int_X,:] = [0, 0, 255]
        cv2.circle(source_image, (int(p_center_x), int(p_center_y)), 2, (0,255,0), -1)
        cv2.imshow("input image", source_image)
        cv2.imwrite("/path/to/save/SSD_Priorbox_Visualization.png", source_image)
        cv2.waitKey(0)
    # break

