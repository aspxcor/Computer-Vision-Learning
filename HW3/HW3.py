# Homework 3 Name: Harris Corner Detection
# Program description:
#Read in camera information and play back the video. Press the space bar once to pause the playback, and perform a Harris Corner detection of the current frame image, and superimpose the detection result on the original image.
#   1. You need to write your own code to implement the Harris Corner detection algorithm, and you cannot directly call the functions related to Harris corner detection in OpenCV;
#   2. Display the intermediate processing results and the final detection results, including the maximum eigenvalue map and the minimum eigenvalue map R map (you can consider color display to superimpose the detection results on the original image, etc., and output these intermediate results as image files.
#   [Node: Please Reference courseware and `lkdemo.c`]

import cv2
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Some global variables and basic hyperparameter information are defined here
Save_Path = "./Output/"     # Path to save picture which is after detected
Window_Size = 3    # Window size of Harris Corner Detect
Harris_Corner_Constant = 0.04      # Harris corner constant. Usually 0.04 - 0.06
Thresh = 10000        # The threshold above which a corner is counted

# When the picture save directory does not exist, create a new directory to save the file
if not os.path.isdir(Save_Path):
    os.makedirs(Save_Path)

# [Function name] gcd
# [Function Usage] This function is used to calculate the greatest common divisor of two Parameter
# [Parameter] two numbers to calculate the greatest common divisor
# [Return value] the calculate the greatest common divisor of two numbers
# [Developer and date] Anonymous
# [Change Record] None
def gcd(a, b):
    if a < b:
        a, b = b, a
    while b != 0:
        temp = a % b
        a = b
        b = temp
    return a

# [Function name] HarrisCornerDetection
# [Function Usage] This function is used to Finsh the Harris Corner Detection
# [Parameter]
    # img: The original image (gray)
    # color_img: The original image (RGB)
    # window_size: The size (side length) of the sliding window
    # k: Harris corner constant. Usually 0.04 - 0.06
    # thresh: The threshold above which a corner is counted
# [Return value] None
# [Developer and date] Anonymous
# [Change Record] None
def HarrisCornerDetection(img, color_img,window_size, k, thresh):
    #Find x and y derivatives
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]
    minEigenvalueImg = np.zeros((height, width, 3), np.uint8)
    minEigenvalueImg[:] = [255,255,255]      # Create a new blank canvas for the minimum eigenvalue map
    maxEigenvalueImg = np.zeros((height, width, 3), np.uint8)
    maxEigenvalueImg[:] = [255,255,255]      # Create a new blank canvas for maximum eigenvalue map
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
            cornerList[x][y]=int(r)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
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
                maxEigenvalueImg.itemset((y, x, 0), 0)
                maxEigenvalueImg.itemset((y, x, 1), 0)
                maxEigenvalueImg.itemset((y, x, 2), 255)
    return color_img, minEigenvalueImg,maxEigenvalueImg,cornerList

# [Function name] drawRMap
# [Function Usage] This function is used to Draw the R graph of the current frame
# [Parameter]
    # data: R value matrix of the current frame
    # count: The sequence number of the current frame
# [Return value] None
# [Developer and date] Anonymous
# [Change Record] None
def drawRMap(data,count):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(0,data.shape[0],int(data.shape[0]/gcd(data.shape[0],data.shape[1])))
    y = np.arange(0,data.shape[1],int(data.shape[1]/gcd(data.shape[0],data.shape[1])))
    X, Y = np.meshgrid(x, y)  # [important] Create grid np.meshgrid(xnums,ynums)
    Z=np.zeros((x.size,y.size))
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j]=data[x[i],y[j]]
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.savefig(Save_Path+'RValue%s.jpg'%str(count))
    cv2.imshow("R Values of Picture after Harris Corner Detection", cv2.imread(Save_Path+'RValue%s.jpg'%str(count)))

# Main program entry
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)   # Call the camera and take a picture
    count=1
    while True:
        ret, frame = cap.read() # Read current frame information
        cv2.namedWindow("Camara Capture", 0)
        cv2.resizeWindow("Camara Capture", 800, 600)
        cv2.resizeWindow("Camara Capture", 800, 600)
        cv2.imshow("Camara Capture", frame)    # Display the current frame captured by the camera
        # Pause and process the current frame when it detects that the user presses the space
        if cv2.waitKey(100) & 0xff == ord(' '):
            imgname="currentFrame%s.jpg"%str(count)
            print("[Frame %s] Current Frame is going to be Detected"%count)
            path = os.path.join(Save_Path, imgname)  # Picture save path
            cv2.imwrite(path,frame)
            img= cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            finalImg, minEigenvalueImg, maxEigenvalueImg, cornerList = HarrisCornerDetection(img,frame, int(Window_Size),float(Harris_Corner_Constant), int(Thresh))
            cv2.imwrite(os.path.join(Save_Path, "finalimage%s.jpg"%str(count)), finalImg)
            cv2.imwrite(os.path.join(Save_Path, "finalminEigenvalueImg%s.jpg"%str(count)), minEigenvalueImg)
            cv2.imwrite(os.path.join(Save_Path, "finalmaxEigenvalueImg%s.jpg"%str(count)), maxEigenvalueImg)
            cv2.imshow("Final Img with Corner Detected in the Picture",finalImg)
            cv2.imshow("Min Eigenvalue Picture after Harris Corner Detection", minEigenvalueImg)
            cv2.imshow("Max Eigenvalue Picture after Harris Corner Detection", maxEigenvalueImg)
            drawRMap(cornerList,count)
            print("[Frame %s] Current Frame has been Detected" % count)
            count += 1
            cv2.waitKey(0)      #break
    cap.release()
    cv2.destroyAllWindows()