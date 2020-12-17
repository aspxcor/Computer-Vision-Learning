# Homework 3 Name: Harris Corner Detection
# Program description:
#Read in camera information and play back the video. Press the space bar once to pause the playback, and perform a Harris Corner detection of the current frame image, and superimpose the detection result on the original image.
#   1. You need to write your own code to implement the Harris Corner detection algorithm, and you cannot directly call the functions related to Harris corner detection in OpenCV;
#   2. Display the intermediate processing results and the final detection results, including the maximum eigenvalue map and the minimum eigenvalue map R map (you can consider color display to superimpose the detection results on the original image, etc., and output these intermediate results as image files.
#   [Node: Please Reference courseware and `lkdemo.c`]
# File Name: HW3_3170104656_DingZhi.py
# Author: Zhi DING
# Student ID: 3170104656
# Last Modified: 2020/12/20

import cv2
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from pyheatmap.heatmap import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Some global variables and basic hyperparameter information are defined here
Path = "./source/"          # Path to save picture which is going to be detected
Save_Path = "./Output/"     # Path to save picture which is after detected
Window_Size = 3    # Window size of Harris Corner Detect
Harris_Corner_Constant = 0.04      # Harris corner constant. Usually 0.04 - 0.06
Thresh = 10000        # The threshold above which a corner is counted


if not os.path.isdir(Save_Path):
    os.makedirs(Save_Path)
# [Class name] Canny
# [Class Usage] This class is used to detect the edge of the image
# [Class Interface]
    # Get_gradient_img(self):Calculate the gradient map and gradient direction matrix and return the generated gradient map
    # Non_maximum_suppression(self):Perform non-maximization suppression on the generated gradient map, combine the magnitude of the tan value with positive and negative, determine the direction of the gradient in the dispersion, and return the generated non-maximization suppression result map
    # Hysteresis_thresholding(self):The hysteresis threshold method is applied to the generated non-maximization suppression result graph, the weak edge is extended with the strong edge, where the extension direction is the vertical direction of the gradient, and the point larger than the low threshold and smaller than the high threshold is set as the high threshold size, direction The determination at the discrete point is similar to the non-maximization suppression, and the result graph of the hysteresis threshold method is returned
    # canny_algorithm(self):Call all the above member functions in order and steps and return Canny edge detection results
# [Developer and date] Zhi DING 2020/12/11
# [Change Record] None


# [Function name] __init__
# [Function Usage] This function is used to Initialize the Canny class
# [Parameter]
    # Guassian_kernal_size: Gaussian filter size
    # img: input picture, changed during the algorithm
    # HT_high_threshold: The high threshold in the hysteresis threshold method
    # HT_low_threshold: The low threshold in the hysteresis threshold method
# [Return value] None
# [Developer and date] Zhi DING 2020/12/11
# [Change Record] None
def findCorners(img, color_img,window_size, k, thresh):
    """
    Finds and returns list of corners and new image with corners drawn
    :param img: The original image
    :param window_size: The size (side length) of the sliding window
    :param k: Harris corner constant. Usually 0.04 - 0.06
    :param thresh: The threshold above which a corner is counted
    :return:
    """
    #Find x and y derivatives
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]

    # cornerList = []
    minEigenvalueImg = np.zeros((height, width, 3), np.uint8)
    minEigenvalueImg[:] = [255,255,255]      # 新建空白画布用于最小特征值图
    maxEigenvalueImg = np.zeros((height, width, 3), np.uint8)
    maxEigenvalueImg[:] = [255,255,255]      # 新建空白画布用于最大特征值图
    # flatImg = np.zeros((height, width, 3), np.uint8)
    # flatImg[:] = [255, 255, 255]  # 新建空白画布用于最大特征值图
    # newImg = img.copy()
    # color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    offset = int(window_size/2)
    cornerList=np.zeros((width,height),dtype=np.int)
    #Loop through image and find our corners
    print ("Finding Corners...")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            #Calculate sum of squares
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)
            # cornerList.append([x, y, r])
            cornerList[x][y]=int(r)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # print(cornerList[x][y])
            #If corner response is over threshold, color the point and add to corner list
            if cornerList[x][y] > thresh:
                if cornerList[x-offset:x+offset+1,y-offset:y+offset+1].max()==cornerList[x][y]:
                    color_img.itemset((y, x, 0), 0)
                    color_img.itemset((y, x, 1), 0)
                    color_img.itemset((y, x, 2), 255)
                    minEigenvalueImg.itemset((y, x, 0), 0)
                    minEigenvalueImg.itemset((y, x, 1), 0)
                    minEigenvalueImg.itemset((y, x, 2), 255)
            elif cornerList[x][y]<0:
                # if cornerList[x - offset:x + offset + 1, y - offset:y + offset + 1].min() == cornerList[x][y]:
                maxEigenvalueImg.itemset((y, x, 0), 0)
                maxEigenvalueImg.itemset((y, x, 1), 0)
                maxEigenvalueImg.itemset((y, x, 2), 255)
            # if abs(cornerList[x][y])<thresh:
            #     flatImg.itemset((y, x, 0), 0)
            #     flatImg.itemset((y, x, 1), 0)
            #     flatImg.itemset((y, x, 2), 255)
    return color_img, minEigenvalueImg,maxEigenvalueImg,cornerList

def apply_heatmap(data,count):
    '''image是原图，data是坐标'''
    '''创建一个新的与原图大小一致的图像，color为0背景为黑色。这里这样做是因为在绘制热力图的时候如果不选择背景图，画出来的图与原图大小不一致（根据点的坐标来的），导致无法对热力图和原图进行加权叠加，因此，这里我新建了一张背景图。'''
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(0,data.shape[0],4)
    y = np.arange(0,data.shape[1],3)
    X, Y = np.meshgrid(x, y)  # [important] 创建网格 np.meshgrid(xnums,ynums)
    Z=np.zeros((x.size,y.size))
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j]=data[x[i],y[j]]
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.savefig('./Output/RValue%s.jpg'%str(count))
    cv2.imshow("R Values of Picture after Harris Corner Detection", cv2.imread('./Output/RValue%s.jpg'%str(count)))


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)   # 参数为视频文件目录
    count=1
    while True:
        ret, frame = cap.read() # 读取
        cv2.namedWindow("Camara Capture", 0)
        cv2.resizeWindow("Camara Capture", 800, 600)
        cv2.imshow("Camara Capture", frame)    # 显示
        if cv2.waitKey(100) & 0xff == ord(' '): # 按空格暂停并处理当前帧
            imgname="currentFrame%s.jpg"%str(count)
            print("[Frame %s] Current Frame has been saved in current folder"%count)
            path = os.path.join(Save_Path, imgname)  # 图片保存路径
            cv2.imwrite(path,frame)

            img= cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            finalImg, minEigenvalueImg, maxEigenvalueImg, cornerList = findCorners(img,frame, int(Window_Size),float(Harris_Corner_Constant), int(Thresh))
            cv2.imwrite(os.path.join(Save_Path, "finalimage%s.jpg"%str(count)), finalImg)
            cv2.imwrite(os.path.join(Save_Path, "finalminEigenvalueImg%s.jpg"%str(count)), minEigenvalueImg)
            cv2.imwrite(os.path.join(Save_Path, "finalmaxEigenvalueImg%s.jpg"%str(count)), maxEigenvalueImg)
            cv2.imshow("Final Img with Corner Detected in the Picture",finalImg)
            cv2.imshow("Min Eigenvalue Picture after Harris Corner Detection", minEigenvalueImg)
            cv2.imshow("Max Eigenvalue Picture after Harris Corner Detection", maxEigenvalueImg)
            # cv2.imwrite(os.path.join(Save_Path, "flatImg%s.jpg"%str(count)), flatImg)

            apply_heatmap(cornerList,count)


            # outfile = open('corners.txt', 'w')
            # outfile.write(corner)
            # outfile.close()

            # for i in range(304964):
            #     outfile.write(str(cornerList[i][0]) + ' ' + str(cornerList[i][1]) + ' ' + str(cornerList[i][2]) + '\n')
            # outfile.close()

            count += 1
            cv2.waitKey(0)      #break
    cap.release()
    cv2.destroyAllWindows()