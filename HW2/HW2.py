# Homework 2 Name: Contour/line extraction
# Program description:
#     1. The core function of the detection algorithm needs to be implemented by writing own code. You cannot call opencv or other functions related to circular line detection in the SDK. If you want to use edge detection, this can be adjusted Use opencv function.
#     2. Display the final test results on the original image;
#     3. Show some key intermediate results separately
#     4. Debugging results of three specified test images (coin, seal and highway) must be carried out. In addition, you can voluntarily add some test images
# File Name: HW2_3170104656_DingZhi.py
# Author: Zhi DING
# Student ID: 3170104656
# Last Modified: 2020/12/6

import cv2
import numpy as np
import time
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
# from multiprocessing import Pool

RADIUSMAX = 130     #

def fill_acc_array(x0, y0, radius):
    x = radius
    y = 0
    decision = 1 - x

    while (y < x):
        if (x + x0 < height and y + y0 < width):
            acc_array[x + x0, y + y0, radius] += 1;  # Octant 1
        if (y + x0 < height and x + y0 < width):
            acc_array[y + x0, x + y0, radius] += 1;  # Octant 2
        if (-x + x0 < height and y + y0 < width):
            acc_array[-x + x0, y + y0, radius] += 1;  # Octant 4
        if (-y + x0 < height and x + y0 < width):
            acc_array[-y + x0, x + y0, radius] += 1;  # Octant 3
        if (-x + x0 < height and -y + y0 < width):
            acc_array[-x + x0, -y + y0, radius] += 1;  # Octant 5
        if (-y + x0 < height and -x + y0 < width):
            acc_array[-y + x0, -x + y0, radius] += 1;  # Octant 6
        if (x + x0 < height and -y + y0 < width):
            acc_array[x + x0, -y + y0, radius] += 1;  # Octant 8
        if (y + x0 < height and -x + y0 < width):
            acc_array[y + x0, -x + y0, radius] += 1;  # Octant 7
        y += 1
        if (decision <= 0):
            decision += 2 * y + 1
        else:
            x = x - 1;
            decision += 2 * (y - x) + 1

def detectCircleLoop(edges,item):
    x = edges[0][item]
    y = edges[1][item]
    for radius in range(20, 130):
        fill_acc_array(x, y, radius)
        print("in detectCircleLoop! i=", item, " len(edges[0])=", len(edges[0]), " radius=", radius)

# def detectCircleLoop(idx):
#     x = edges[0][idx]
#     y = edges[1][idx]
#     for radius in range(20, 130):
#         fill_acc_array(x, y, radius)
#         print("in detectCircleLoop! i=", idx, " len(edges[0])=", len(edges[0]), " radius=", radius)

def detectCircle():
    filter3D = np.ones((30, 30, RADIUSMAX))
    edges = np.where(edged_image == 255)
    # pool = ThreadPool(8) # 池的大小为8
    pool = ThreadPool(8)  # 池的大小为8
    detectCircleLoopFunction=partial(detectCircleLoop,edges)
    pool.map(detectCircleLoopFunction,range(0, len(edges[0])))

    # pool.map(detectCircleLoop,range(0, len(edges[0])))

    # map(detectCircleLoop,range(0, len(edges[0])))

    # for i in range(0, len(edges[0])):
    #     x = edges[0][i]
    #     y = edges[1][i]
    #     for radius in range(20, 130):
    #         fill_acc_array(x, y, radius)
    #         print("in detecting loop! i=", i, " len(edges[0])=", len(edges[0]), " radius=", radius)
    # i = 0
    # j = 0
    # while (i < height - 30):
    for i in range(0,height - 30,30):
        # while (j < width - 30):
        for j in range(0, width - 30, 30):
            print("in detecting loop: i=", i, " height-30=", height - 30, " j=", j, " width-30=", width - 30)
            filter3D = acc_array[i:i + 30, j:j + 30, :] * filter3D
            max_pt = np.where(filter3D == filter3D.max())
            a = max_pt[0]
            b = max_pt[1]
            c = max_pt[2]
            b = b + j
            a = a + i
            print("a=", a)
            print("b=", b)
            print("c=", c)
            if (filter3D.max() > 150):
                # print("in if")
                cv2.circle(output, (b[0], a[0]), c[0], (0, 255, 0), 2)
            # j = j + 30
            filter3D[:, :, :] = 1
        # j = 0
        # i = i + 30
    cv2.imshow('Detected circle', output)

if __name__ == "__main__":
    original_image = cv2.imread('hw-coin.jpg')
    output = original_image.copy()
    #Gaussian Blurring of Gray Image
    blur_image = cv2.GaussianBlur(original_image,(23,23),0)
    #Using OpenCV Canny Edge detector to detect edges
    edged_image = cv2.Canny(blur_image,75,150)
    cv2.imshow('Original Image',original_image)
    cv2.imshow('Gaussian Blurred Image',blur_image)
    cv2.imshow('Edged Image', edged_image)
    height,width = edged_image.shape
    # radii = 130

    acc_array = np.zeros(((height,width,RADIUSMAX)))
    # edges = np.where(edged_image == 255)

    start_time = time.time()

    detectCircle()

    end_time = time.time()
    time_taken = end_time - start_time
    print ('Time taken for execution',time_taken)



    cv2.waitKey(0)
    cv2.destroyAllWindows()


# import os
# import cv2
# import numpy as np
#
# img = cv2.imread("./source/hw-seal.jpg")
# cv2.imshow("original_img", img)
#
# # canny(): 边缘检测
# img1 = cv2.GaussianBlur(img,(3,3),0)
# canny = cv2.Canny(img1, 50, 150)
# cv2.imshow('Canny', canny)
#
#
# # _,Thr_img = cv2.threshold(img,210,255,cv2.THRESH_BINARY)#设定红色通道阈值210（阈值影响梯度运算效果）
# # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))         #定义矩形结构元素
# # gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel) #梯度
# # cv2.imshow("gradient", gradient)
#
# # cv2.imwrite("canny.jpg", cv2.Canny(img, 200, 300))
# # cv2.imshow("canny", cv2.imread("canny.jpg"))
#
# cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import sys
# from os.path import splitext
#
#
# def crop(image, r, c, height, width):
#     return image[r:r+height, c:c+width]
#
#
# def moore_neighbor_tracing(image, accumulator, maxr):
#     color = 100
#     original_height, original_width = image.shape
#     image = np.pad(image, ((1, 1), (1, 1)), 'constant', constant_values=(255, 255))
#     height, width = image.shape
#     contour_pixels = []
#     p = (0, 0)
#     c = (0, 0)
#     s = (0, 0)
#     previous = (0, 0)
#     found = False
#
#     # Find the first point
#     for i in range(height):
#         for j in range(width):
#             if image[i, j] <= color and not (i == 0 and j == 0):
#                 s = (i, j)
#                 # contour_pixels.append(s)
#                 contour_pixels.append((s[0]-1, s[1]-1))
#                 p = s
#                 found = True
#                 break
#             if not found:
#                 previous = (i, j)
#         if found:
#             break
#
#     # If the pixel is isolated i don't do anything
#     isolated = True
#     m = moore_neighbor(p)
#     for r, c in m:
#         if image[r, c] <= color:
#             isolated = False
#
#     if not isolated:
#         tmp = c
#         # Backtrack and next clockwise M(p)
#         c = next_neighbor(s, previous)
#         previous = tmp
#         while c != s:
#             if image[c] <= color:
#                 previous_contour = contour_pixels[len(contour_pixels) - 1]
#
#                 # contour_pixels.append(c)
#                 contour_pixels.append((c[0]-1, c[1]-1))
#                 p = c
#                 c = previous
#
#                 # HERE is where i have to start checking for lines
#                 # i get the previous contour pixel
#                 current_contour = p[0] - 1, p[1] - 1
#
#                 # i have to calculate t (between 0 and 179) of the line that connects the two pixel
#                 t = np.arctan2(previous_contour[1]-current_contour[1], previous_contour[0]-current_contour[0]) * 180 / np.pi
#                 t = int(np.round(t))
#                 if t < 0:
#                     t += 180
#                 t %= 180
#
#                 # This is the "classic" Hough in which we consider only a subset of all the possible lines
#                 for t in range(t-30, t+31):
#                     if t >= 180:
#                         t = 180 - t
#                     if t < 0:
#                         t = 180 + t
#                     rad = np.deg2rad(t)
#
#                     r = current_contour[0] * np.sin(rad) + current_contour[1] * np.cos(rad) + maxr
#                     accumulator[int(np.round(r)), t] += 1
#
#             else:
#                 previous = c
#                 c = next_neighbor(p, c)
#
#         image = crop(image, 1, 1, original_height, original_width)
#     return contour_pixels
#
#
# def moore_neighbor(pixel):
#     row, col = pixel
#     return ((row - 1, col - 1), (row - 1, col), (row - 1, col + 1),
#             (row, col + 1), (row + 1, col + 1),
#             (row + 1, col), (row + 1, col - 1),
#             (row, col - 1))
#
#
# def next_neighbor(central, neighbor):
#     neighbors = moore_neighbor(central)
#     index = np.where((np.array(neighbors) == neighbor).all(axis=1))[0][0]
#     index += 1
#     index = index % 8
#
#     # Problem operating like this:
#     # if the object of which i want to detect contours starts at the edges of the image there's the possibility
#     # of going out of bounds
#     return neighbors[index]
#
#
# # Function that "deletes" an object using the information about its contours
# def delete_object(image, contoured, contours):
#     # With the edge pixel i also delete its moore neighborhood because otherwise if and edge is 2 pixel thick
#     # because i find only the external contour i wouldn't delete the contour completely
#     height, width = image.shape
#     for x, y in contours:
#         image[x, y] = 255
#         image[np.clip(x - 1, 0, height - 1), np.clip(y - 1, 0, width - 1)] = 255
#         image[np.clip(x - 1, 0, height - 1), y] = 255
#         image[np.clip(x - 1, 0, height - 1), np.clip(y + 1, 0, width - 1)] = 255
#         image[x, np.clip(y - 1, 0, width - 1)] = 255
#         image[x, y] = 255
#         image[x, np.clip(y + 1, 0, width - 1)] = 255
#         image[np.clip(x + 1, 0, height - 1), np.clip(y - 1, 0, width - 1)] = 255
#         image[np.clip(x + 1, 0, height - 1), y] = 255
#         image[np.clip(x + 1, 0, height - 1), np.clip(y + 1, 0, width - 1)] = 255
#
#     return image
#
#
# # INPUT PARAMETER: a binarized image (using Canny edge) in which the contours are black and the background is white
# # OUTPUT PARAMETER: an image with black background and white contours for all the objects identified and red lines
# #                   where they are found
# def main():
#     if len(sys.argv) > 1:
#         # Used later to save the image
#         image_basename = splitext(sys.argv[1])[0]
#         img_format = splitext(sys.argv[1])[1]
#
#         # Load the image
#         image = cv2.imread(sys.argv[1], 0)
#         contour_all = np.zeros(image.shape, np.uint8)
#
#         height, width = image.shape
#         # Variables used to perform hough
#         maxr = int(np.ceil(np.sqrt(np.power(height, 2) + np.power(width, 2))))
#         accumulator = np.zeros((maxr * 2 + 1, 180), np.uint32)
#
#         # I iterate until there are no more edge pixels
#         # I keep tracing contours updating the accumulator matrix
#         # delete the contour found and repeat
#         while np.any(image <= 100):
#             contoured = np.zeros(image.shape, np.uint8)
#             contours = moore_neighbor_tracing(image, accumulator, maxr)
#             for x, y in contours:
#                 contoured[x, y] = 255
#                 contour_all[x, y] = 255
#             delete_object(image, contoured, contours)
#
#         cv2.imshow("", contour_all)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         cv2.imwrite(image_basename + "_contoured" + img_format, contour_all)
#
#         # Draw the lines i find in the accumulator matrix
#         tmp = np.array(contour_all)
#         print("before merge")
#         tmp = cv2.merge((tmp, tmp, tmp))
#         print("after merge")
#         t = 100
#         for i, j in np.argwhere(accumulator > t):
#             print("in loop1,i=",i," ,j=",j)
#             rad = np.deg2rad(j)
#             a = np.cos(rad)
#             b = np.sin(rad)
#             # This is needed because debugging i saw that a or b could go something like 1e-17 which is 0
#             # but if i don't set it manually to 0, due to approximation errors i could get that the two points
#             # of an horizontal line have two y that differs by one unit and so the line is not perfectly horizontal
#             # (it happened with the sudoku image, for example)
#             if 0 < a < 1e-10:
#                 a = 0
#             if 0 < b < 1e-10:
#                 b = 0
#             x0 = a * (i - maxr)
#             y0 = b * (i - maxr)
#             x1 = int(x0 + 1000 * (-b))
#             y1 = int(y0 + 1000 * (a))
#             x2 = int(x0 - 1000 * (-b))
#             y2 = int(y0 - 1000 * (a))
#             cv2.line(tmp, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
#
#         # Post processing: i delete red lines where there is not a white line beneath
#         height, width = image.shape
#         for i in range(height):
#             for j in range(width):
#                 print("in loop2,i=",i,",j=",j,",height=",height,"width=",width)
#                 # if a pixel is red i check if it is correct, otherwise i set it to black
#                 # to check if it is correct i look if under it there are white pixels in the moor neighborhood
#                 if tmp[i, j, 2] == 255:
#                     mn = moore_neighbor((i, j))
#                     mn = np.array(mn)
#                     rows = np.clip(mn[:, 0], 0, height-1)
#                     columns = np.clip(mn[:, 1], 0, width-1)
#                     colored_neighbor = np.any(contour_all[(rows, columns)] >= 200)
#                     if contour_all[i, j] <= 100 and not colored_neighbor:
#                         tmp[i, j, 2] = 0
#
#         cv2.imshow("", tmp)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         cv2.imwrite(image_basename + "_houghed2" + img_format, tmp)
#
#     else:
#         print("Not enough input arguments")
#
#
# if __name__ == "__main__":
#     main()



# import numpy as np
# import imageio
# import math
#
# def rgb2gray(rgb):
#     return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
#
#
# def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
#     """
#     Hough transform for lines
#     Input:
#     img - 2D binary image with nonzeros representing edges
#     angle_step - Spacing between angles to use every n-th angle
#                  between -90 and 90 degrees. Default step is 1.
#     lines_are_white - boolean indicating whether lines to be detected are white
#     value_threshold - Pixel values above or below the value_threshold are edges
#     Returns:
#     accumulator - 2D array of the hough transform accumulator
#     theta - array of angles used in computation, in radians.
#     rhos - array of rho values. Max size is 2 times the diagonal
#            distance of the input image.
#     """
#     # Rho and Theta ranges
#     thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
#     width, height = img.shape
#     diag_len = int(round(math.sqrt(width * width + height * height)))
#     rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
#
#     # Cache some resuable values
#     cos_t = np.cos(thetas)
#     sin_t = np.sin(thetas)
#     num_thetas = len(thetas)
#
#     # Hough accumulator array of theta vs rho
#     accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
#     # (row, col) indexes to edges
#     are_edges = img > value_threshold if lines_are_white else img < value_threshold
#     y_idxs, x_idxs = np.nonzero(are_edges)
#
#     # Vote in the hough accumulator
#     for i in range(len(x_idxs)):
#         x = x_idxs[i]
#         y = y_idxs[i]
#
#         for t_idx in range(num_thetas):
#             print("i:",i," ,x:",x," ,y:",y," ,t_idx:",t_idx)
#             # Calculate rho. diag_len is added for a positive index
#             rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
#             accumulator[rho, t_idx] += 1
#
#     return accumulator, thetas, rhos
#
#
# def show_hough_line(img, accumulator, save_path=None):
#     import matplotlib.pyplot as plt
#
#     fig, ax = plt.subplots(1, 2, figsize=(10, 10))
#
#     ax[0].imshow(img, cmap=plt.cm.gray)
#     ax[0].set_title('Input image')
#     ax[0].axis('image')
#
#     ax[1].imshow(
#         accumulator, cmap='jet',
#         extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
#     ax[1].set_aspect('equal', adjustable='box')
#     ax[1].set_title('Hough transform')
#     ax[1].set_xlabel('Angles (degrees)')
#     ax[1].set_ylabel('Distance (pixels)')
#     ax[1].axis('image')
#
#     # plt.axis('off')
#     if save_path is not None:
#         plt.savefig(save_path, bbox_inches='tight')
#     plt.show()
#
#
# if __name__ == '__main__':
#     imgpath = "./source/hw-seal.jpg"
#     img = imageio.imread(imgpath)
#     if img.ndim == 3:
#         img = rgb2gray(img)
#     accumulator, thetas, rhos = hough_line(img)
#     show_hough_line(img, accumulator, save_path="./source/output.jpg")


# from scipy import misc
#
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# import imageio
#
# #----------------------------------------------------------------------------------------#
# # Step 1: read image
#
# img = imageio.imread("./source/hw-seal.jpg")
#
# print ('image shape: ', img.shape)
#
# plt.imshow(img, )
#
# plt.savefig("image.png",bbox_inches='tight')
#
# plt.close()
#
# #----------------------------------------------------------------------------------------#
# # Step 2: Hough Space
#
# img_shape = img.shape
#
# x_max = img_shape[0]
# y_max = img_shape[1]
#
# theta_max = 1.0 * math.pi
# theta_min = 0.0
#
# r_min = 0.0
# r_max = math.hypot(x_max, y_max)
#
# r_dim = 200
# theta_dim = 300
#
# hough_space = np.zeros((r_dim,theta_dim))
#
# for x in range(x_max):
#     for y in range(y_max):
#         if img[x,y,0] == 255: continue
#         for itheta in range(theta_dim):
#             print("X:",x," Y:",y," itheta:",itheta)
#             theta = 1.0 * itheta * theta_max / theta_dim
#             r = x * math.cos(theta) + y * math.sin(theta)
#             ir = int(r_dim * ( 1.0 * r ) / r_max)
#             hough_space[ir,itheta] = hough_space[ir,itheta] + 1
#
# plt.imshow(hough_space, origin='lower')
# plt.xlim(0,theta_dim)
# plt.ylim(0,r_dim)
#
# tick_locs = [i for i in range(0,theta_dim,40)]
# tick_lbls = [round( (1.0 * i * theta_max) / theta_dim,1) for i in range(0,theta_dim,40)]
# plt.xticks(tick_locs, tick_lbls)
#
# tick_locs = [i for i in range(0,r_dim,20)]
# tick_lbls = [round( (1.0 * i * r_max ) / r_dim,1) for i in range(0,r_dim,20)]
# plt.yticks(tick_locs, tick_lbls)
#
# plt.xlabel(r'Theta')
# plt.ylabel(r'r')
# plt.title('Hough Space')
#
# plt.savefig("hough_space_r_theta.png",bbox_inches='tight')
#
# plt.close()
#
# #----------------------------------------------------------------------------------------#
# # Find maximas 1
# '''
# Sorted_Index_HoughTransform =  np.argsort(hough_space, axis=None)
#
# print 'Sorted_Index_HoughTransform[0]', Sorted_Index_HoughTransform[0]
# #print Sorted_Index_HoughTransform.shape, r_dim * theta_dim
#
# shape = Sorted_Index_HoughTransform.shape
#
# k = shape[0] - 1
# list_r = []
# list_theta = []
# for d in range(5):
#     i = int( Sorted_Index_HoughTransform[k] / theta_dim )
#     #print i, round( (1.0 * i * r_max ) / r_dim,1)
#     list_r.append(round( (1.0 * i * r_max ) / r_dim,1))
#     j = Sorted_Index_HoughTransform[k] - theta_dim * i
#     print 'Maxima', d+1, 'r: ', j, 'theta', round( (1.0 * j * theta_max) / theta_dim,1)
#     list_theta.append(round( (1.0 * j * theta_max) / theta_dim,1))
#     print "--------------------"
#     k = k - 1
#
#
# #theta = list_theta[7]
# #r = list_r[7]
#
# #print " r,theta",r,theta, math.degrees(theta)
# '''
# #----------------------------------------------------------------------------------------#
# # Step 3: Find maximas 2
#
# import scipy.ndimage.filters as filters
# import scipy.ndimage as ndimage
#
# neighborhood_size = 20
# threshold = 140
#
# data_max = filters.maximum_filter(hough_space, neighborhood_size)
# maxima = (hough_space == data_max)
#
#
# data_min = filters.minimum_filter(hough_space, neighborhood_size)
# diff = ((data_max - data_min) > threshold)
# maxima[diff == 0] = 0
#
# labeled, num_objects = ndimage.label(maxima)
# slices = ndimage.find_objects(labeled)
#
# x, y = [], []
# for dy,dx in slices:
#     x_center = (dx.start + dx.stop - 1)/2
#     x.append(x_center)
#     y_center = (dy.start + dy.stop - 1)/2
#     y.append(y_center)
#
# print (x)
# print (y)
#
# plt.imshow(hough_space, origin='lower')
# plt.savefig('hough_space_i_j.png', bbox_inches = 'tight')
#
# plt.autoscale(False)
# plt.plot(x,y, 'ro')
# plt.savefig('hough_space_maximas.png', bbox_inches = 'tight')
#
# plt.close()
#
# #----------------------------------------------------------------------------------------#
# # Step 4: Plot lines
#
# line_index = 1
#
# for i,j in zip(y, x):
#
#     r = round( (1.0 * i * r_max ) / r_dim,1)
#     theta = round( (1.0 * j * theta_max) / theta_dim,1)
#
#     fig, ax = plt.subplots()
#
#     ax.imshow(img)
#
#     ax.autoscale(False)
#
#     px = []
#     py = []
#     for i in range(-y_max-40,y_max+40,1):
#         px.append( math.cos(-theta) * i - math.sin(-theta) * r )
#         py.append( math.sin(-theta) * i + math.cos(-theta) * r )
#
#     ax.plot(px,py, linewidth=10)
#
#     plt.savefig("image_line_"+ "%02d" % line_index +".png",bbox_inches='tight')
#
#     #plt.show()
#
#     plt.close()
#
#     line_index = line_index + 1
#
# #----------------------------------------------------------------------------------------#
# # Plot lines
# '''
# i = 11
# j = 264
#
# i = y[1]
# j = x[1]
#
# print i,j
#
# r = round( (1.0 * i * r_max ) / r_dim,1)
# theta = round( (1.0 * j * theta_max) / theta_dim,1)
#
# print 'r', r
# print 'theta', theta
#
#
# fig, ax = plt.subplots()
#
# ax.imshow(img)
#
# ax.autoscale(False)
#
# px = []
# py = []
# for i in range(-y_max-40,y_max+40,1):
#     px.append( math.cos(-theta) * i - math.sin(-theta) * r )
#     py.append( math.sin(-theta) * i + math.cos(-theta) * r )
#
# print px
# print py
#
# ax.plot(px,py, linewidth=10)
#
# plt.savefig("PlottedLine_07.png",bbox_inches='tight')
#
# #plt.show()
#
# '''
