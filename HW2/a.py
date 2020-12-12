# Homework 2 Name: Contour/line extraction
# Program description:
#     1. The core function of the detection algorithm needs to be implemented by writing own code. You cannot call opencv or other functions related to circular line detection in the SDK. If you want to use edge detection, this can be adjusted Use opencv function.
#     2. Display the final test results on the original image;
#     3. Show some key intermediate results separately
#     4. Debugging results of three specified test images (coin, seal and highway) must be carried out. In addition, you can voluntarily add some test images
# File Name: HW2_3170104656_DingZhi.py
# Author: Zhi DING
# Student ID: 3170104656
# Last Modified: 2020/12/11

import cv2
import math
import numpy as np

# Some global variables and basic hyperparameter information are defined here
Path = "./source/"          # Path to save picture which is going to be detected
Save_Path = "./Output/"     # Path to save picture which is after detected
Guassian_kernal_size = 3    # Convolution kernel of GaussianBlur
HT_high_threshold = 25      # high threshold of Canny
HT_low_threshold = 6        # low threshold of Canny
Hough_transform_step = 6    # Step in Hough Transform
Hough_transform_threshold = 110 # threshold of Hough Transform

# [Class name] Canny
# [Class Usage] This class is used to detect the edge of the image
# [Class Interface]
    # Get_gradient_img(self):Calculate the gradient map and gradient direction matrix and return the generated gradient map
    # Non_maximum_suppression(self):Perform non-maximization suppression on the generated gradient map, combine the magnitude of the tan value with positive and negative, determine the direction of the gradient in the dispersion, and return the generated non-maximization suppression result map
    # Hysteresis_thresholding(self):The hysteresis threshold method is applied to the generated non-maximization suppression result graph, the weak edge is extended with the strong edge, where the extension direction is the vertical direction of the gradient, and the point larger than the low threshold and smaller than the high threshold is set as the high threshold size, direction The determination at the discrete point is similar to the non-maximization suppression, and the result graph of the hysteresis threshold method is returned
    # canny_algorithm(self):Call all the above member functions in order and steps and return Canny edge detection results
# [Developer and date] Zhi DING 2020/12/11
# [Change Record] None
class Canny:

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
    def __init__(self, Guassian_kernal_size, img, HT_high_threshold, HT_low_threshold):
        self.Guassian_kernal_size = Guassian_kernal_size
        self.img = img
        self.y, self.x = img.shape[0:2]
        self.angle = np.zeros([self.y, self.x])
        self.img_origin = None
        self.x_kernal = np.array([[-1, 1]])
        self.y_kernal = np.array([[-1], [1]])
        self.HT_high_threshold = HT_high_threshold
        self.HT_low_threshold = HT_low_threshold

    # [Function name] Get_gradient_img
    # [Function Usage] This function is used to Calculate gradient map and gradient direction matrix
    # [Parameter] None
    # [Return value] Return the Generated gradient map
    # [Developer and date] Zhi DING 2020/12/11
    # [Change Record] None
    def Get_gradient_img(self):
        print('Get_gradient_img')
        # Initializes the correlation matrix used to calculate the gradient
        new_img_x = np.zeros([self.y, self.x], dtype=np.float)
        new_img_y = np.zeros([self.y, self.x], dtype=np.float)
        # Scan the image circularly to calculate the gradient, and record the relevant data into the gradient matrix
        for i in range(0, self.x):
            for j in range(0, self.y):
                if j == 0:
                    new_img_y[j][i] = 1
                else:
                    new_img_y[j][i] = np.sum(np.array([[self.img[j - 1][i]], [self.img[j][i]]]) * self.y_kernal)
                if i == 0:
                    new_img_x[j][i] = 1
                else:
                    new_img_x[j][i] = np.sum(np.array([self.img[j][i - 1], self.img[j][i]]) * self.x_kernal)
        # Return amplitude and phase
        gradient_img, self.angle = cv2.cartToPolar(new_img_x, new_img_y)
        self.angle = np.tan(self.angle)
        self.img = gradient_img.astype(np.uint8)
        return self.img

    # [Function name] Non_maximum_suppression
    # [Function Usage] This function is used to Perform non-maximization suppression on the generated gradient map, and combine the tan value with positive and negative to determine the direction of the gradient in the dispersion
    # [Parameter] None
    # [Return value] Return the resulting graph of non-maximization suppression results
    # [Developer and date] Zhi DING 2020/12/11
    # [Change Record] None
    def Non_maximum_suppression(self):
        print('Non_maximum_suppression')
        # Initialize the matrix related to non-maximum suppression
        result = np.zeros([self.y, self.x])
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                # Perform non-maximum suppression in different scenarios
                if abs(self.img[i][j]) <= 4:
                    result[i][j] = 0
                    continue
                elif abs(self.angle[i][j]) > 1:
                    gradient2 = self.img[i - 1][j]
                    gradient4 = self.img[i + 1][j]
                    if self.angle[i][j] > 0:
                        gradient1 = self.img[i - 1][j - 1]
                        gradient3 = self.img[i + 1][j + 1]
                    else:
                        gradient1 = self.img[i - 1][j + 1]
                        gradient3 = self.img[i + 1][j - 1]
                else:
                    gradient2 = self.img[i][j - 1]
                    gradient4 = self.img[i][j + 1]
                    if self.angle[i][j] > 0:
                        gradient1 = self.img[i - 1][j - 1]
                        gradient3 = self.img[i + 1][j + 1]
                    else:
                        gradient3 = self.img[i - 1][j + 1]
                        gradient1 = self.img[i + 1][j - 1]
                # Process the image matrix according to the result of non-maximum suppression
                temp1 = abs(self.angle[i][j]) * gradient1 + (1 - abs(self.angle[i][j])) * gradient2
                temp2 = abs(self.angle[i][j]) * gradient3 + (1 - abs(self.angle[i][j])) * gradient4
                if self.img[i][j] >= temp1 and self.img[i][j] >= temp2:
                    result[i][j] = self.img[i][j]
                else:
                    result[i][j] = 0
        self.img = result
        return self.img

    # [Function name] Hysteresis_thresholding
    # [Function Usage] The hysteresis threshold method is applied to the generated non-maximization suppression result graph, and the weak edge is extended with the strong edge. The extension direction here is the vertical direction of the gradient. The determination at discrete points is similar to non-maximization suppression.
    # [Parameter] None
    # [Return value] Return the Hysteresis threshold method result graph
    # [Developer and date] Zhi DING 2020/12/11
    # [Change Record] None
    def Hysteresis_thresholding(self):
        print('Hysteresis_thresholding')
        # Perform Hysteresis thresholding in different scenarios
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if self.img[i][j] >= self.HT_high_threshold:
                    if abs(self.angle[i][j]) < 1:
                        if self.img_origin[i - 1][j] > self.HT_low_threshold:
                            self.img[i - 1][j] = self.HT_high_threshold
                        if self.img_origin[i + 1][j] > self.HT_low_threshold:
                            self.img[i + 1][j] = self.HT_high_threshold
                        if self.angle[i][j] < 0:
                            if self.img_origin[i - 1][j - 1] > self.HT_low_threshold:
                                self.img[i - 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j + 1] = self.HT_high_threshold
                        else:
                            if self.img_origin[i - 1][j + 1] > self.HT_low_threshold:
                                self.img[i - 1][j + 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j - 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
                    else:
                        if self.img_origin[i][j - 1] > self.HT_low_threshold:
                            self.img[i][j - 1] = self.HT_high_threshold
                        if self.img_origin[i][j + 1] > self.HT_low_threshold:
                            self.img[i][j + 1] = self.HT_high_threshold
                        if self.angle[i][j] < 0:
                            if self.img_origin[i - 1][j - 1] > self.HT_low_threshold:
                                self.img[i - 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j + 1] = self.HT_high_threshold
                        else:
                            if self.img_origin[i - 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j - 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
        return self.img

    # [Function name] canny_algorithm
    # [Function Usage] This function is used to Call all the above member functions in order and steps
    # [Parameter] None
    # [Return value] Return the Results of Canny algorithm
    # [Developer and date] Zhi DING 2020/12/11
    # [Change Record] None
    def canny_algorithm(self):
        # Call all the above member functions in order and steps
        self.img = cv2.GaussianBlur(self.img, (self.Guassian_kernal_size, self.Guassian_kernal_size), 0)
        self.Get_gradient_img()
        self.img_origin = self.img.copy()
        self.Non_maximum_suppression()
        self.Hysteresis_thresholding()
        return self.img

# [Class name] Hough_Circle_Transform
# [Class Usage] This class is used to detect the circle in the image by using the Hough Transform Algorithm
# [Class Interface]
    # Hough_transform_algorithm(self):A three-dimensional space is established according to x, y, and radius, and all units in the space are voted along the gradient direction according to the points on the edge of the picture. The result of each point voted as a broken line and returned to the voting matrix
    # Select_Circle(self):Select suitable circles from the voting matrix according to the threshold, and use the method of averaging the results of neighboring points to suppress non-maximization
    # Calculate(self):Call the above member functions in the order of the algorithm and return the circle fitting result graph, the circle coordinates and radius set
# [Developer and date] Zhi DING 2020/12/11
# [Change Record] None
class Hough_Circle_Transform:

    # [Function name] __init__
    # [Function Usage] This function is used to Initialize the Hough_Circle_Transform class
    # [Parameter]
        # img: input image
        # angle: input gradient direction matrix
        # step: Hough transform step size
        # threshold: the threshold of the filter unit
    # [Return value] None
    # [Developer and date] Zhi DING 2020/12/11
    # [Change Record] None
    def __init__(self, img, angle, step=5, threshold=135):
        self.img = img
        self.angle = angle
        self.y, self.x = img.shape[0:2]
        self.radius = math.ceil(math.sqrt(self.y ** 2 + self.x ** 2))
        self.step = step
        self.vote_matrix = np.zeros(
            [math.ceil(self.y / self.step), math.ceil(self.x / self.step), math.ceil(self.radius / self.step)])
        self.threshold = threshold
        self.circles = []

    # [Function name] Hough_transform_algorithm
    # [Function Usage] A three-dimensional space is established according to x, y, and radius, and all units in the space are voted along the gradient direction according to the points on the edge of the picture. Each point is cast and the result is a broken line.
    # [Parameter] None
    # [Return value] Return the voting matrix
    # [Developer and date] Zhi DING 2020/12/11
    # [Change Record] None
    def Hough_transform_algorithm(self):
        print('Hough_transform_algorithm')
        # A three-dimensional space is established according to x, y, and radius
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if self.img[i][j] > 0:
                    y = i
                    x = j
                    r = 0
                    while y < self.y and x < self.x and y >= 0 and x >= 0:
                        self.vote_matrix[math.floor(y / self.step)][math.floor(x / self.step)][
                            math.floor(r / self.step)] += 1
                        y = y + self.step * self.angle[i][j]
                        x = x + self.step
                        r = r + math.sqrt((self.step * self.angle[i][j]) ** 2 + self.step ** 2)
                    y = i - self.step * self.angle[i][j]
                    x = j - self.step
                    r = math.sqrt((self.step * self.angle[i][j]) ** 2 + self.step ** 2)
                    while y < self.y and x < self.x and y >= 0 and x >= 0:
                        self.vote_matrix[math.floor(y / self.step)][math.floor(x / self.step)][
                            math.floor(r / self.step)] += 1
                        y = y - self.step * self.angle[i][j]
                        x = x - self.step
                        r = r + math.sqrt((self.step * self.angle[i][j]) ** 2 + self.step ** 2)
        return self.vote_matrix

    # [Function name] Select_Circle
    # [Function Usage] Select suitable circles from the voting matrix according to the threshold, and use the method of averaging the results of neighboring points to suppress non-maximization.
    # [Parameter] None
    # [Return value] None
    # [Developer and date] Zhi DING 2020/12/11
    # [Change Record] None
    def Select_Circle(self):
        print('Select_Circle')
        circleCandidate = []        # Store candidate circle information
        for i in range(0, math.ceil(self.y / self.step)):
            for j in range(0, math.ceil(self.x / self.step)):
                for r in range(0, math.ceil(self.radius / self.step)):
                    if self.vote_matrix[i][j][r] >= self.threshold:
                        y = i * self.step + self.step / 2
                        x = j * self.step + self.step / 2
                        r = r * self.step + self.step / 2
                        circleCandidate.append((math.ceil(x), math.ceil(y), math.ceil(r)))
        if len(circleCandidate) == 0:
            print("No Circle in this threshold. No Circle Detected.")
            return
        # Check the candidate circles one by one and deal with them
        x, y, r = circleCandidate[0]
        possible = []
        middle = []
        for circle in circleCandidate:
            if abs(x - circle[0]) <= 20 and abs(y - circle[1]) <= 20:
                possible.append([circle[0], circle[1], circle[2]])
            else:
                result = np.array(possible).mean(axis=0)
                middle.append((result[0], result[1], result[2]))
                possible.clear()
                x, y, r = circle
                possible.append([x, y, r])
        result = np.array(possible).mean(axis=0)
        middle.append((result[0], result[1], result[2]))

        def takeFirst(elem):
            return elem[0]
        # Output candidate circle information
        middle.sort(key=takeFirst)
        x, y, r = middle[0]
        possible = []
        for circle in middle:
            if abs(x - circle[0]) <= 20 and abs(y - circle[1]) <= 20:
                possible.append([circle[0], circle[1], circle[2]])
            else:
                result = np.array(possible).mean(axis=0)
                print("Circle candidate core: (%f, %f) Radius: %f" % (result[0], result[1], result[2]))
                self.circles.append((result[0], result[1], result[2]))
                possible.clear()
                x, y, r = circle
                possible.append([x, y, r])
        result = np.array(possible).mean(axis=0)
        print("Circle candidate core: (%f, %f) Radius: %f" % (result[0], result[1], result[2]))
        self.circles.append((result[0], result[1], result[2]))

    # [Function name] Calculate
    # [Function Usage] Call the above member functions in the order of the algorithm
    # [Parameter] None
    # [Return value] Return the Circle fitting result graph, circle coordinates and radius collection
    # [Developer and date] Zhi DING 2020/12/11
    # [Change Record] None
    def Calculate(self):
        self.Hough_transform_algorithm()
        self.Select_Circle()
        return self.circles

# [Class name] Hough_Line_Transform
# [Class Usage] This class is used to detect the line in the image
# [Class Interface]
    # voting(self):According to the relevant design of the Hough algorithm, the voting matrix is given for the subsequent determination of the straight line position
    # non_maximum_suppression(self):Perform non-maximization suppression on the generated gradient map, and return the generated non-maximization suppression result map
    # inverse_hough(self):According to the previously obtained matrix information and threshold, determine the line position and draw a graph with line detection information
    # Calculate(self):Call all the above member functions in order and steps and return the detection results
# [Developer and date] Zhi DING 2020/12/11
# [Change Record] None
class Hough_Line_Transform:

    # [Function name] __init__
    # [Function Usage] This function is used to Initialize the Hough_Line_Transform class
    # [Parameter]
        # img: input image(gray)
        # imgOrigin: input image(RGB)
    # [Return value] None
    # [Developer and date] Zhi DING 2020/12/11
    # [Change Record] None
    def __init__(self, img, imgOrigin):
        self.img = img
        self.imgOrigin =imgOrigin
        self.y, self.x = img.shape[0:2]
        self.rho_max = np.ceil(np.sqrt(self.y ** 2 + self.x ** 2)).astype(np.int)    # get rho max length
        self.vote_matrix = np.zeros((self.rho_max, 180), dtype=np.int)   # hough table
        self.idx = np.where(self.img == 255)

    # [Function name] voting
    # [Function Usage] According to the relevant design of the Hough algorithm
    # [Parameter] None
    # [Return value] Return the voting matrix
    # [Developer and date] Zhi DING 2020/12/11
    # [Change Record] None
    def voting(self):
        print('voting')
        for y, x in zip(self.idx[0], self.idx[1]):      # zip function returns tuple
            for theta in range(180):
                t = np.pi / 180 * theta     # get polar coordinat4s
                rho = int(x * np.cos(t) + y * np.sin(t))
                self.vote_matrix[rho, theta] += 1       # Vote
        self.vote_matrix=self.vote_matrix.astype(np.uint8)
        return self.vote_matrix.astype(np.uint8)

    # [Function name] inverse_hough
    # [Function Usage] According to the previously obtained matrix information and threshold, determine the line position and draw a graph with line detection information
    # [Parameter] None
    # [Return value] Return the graph with line detection information
    # [Developer and date] Zhi DING 2020/12/11
    # [Change Record] None
    def inverse_hough(self):
        print('inverse_hough')
        rho_max, _ = self.vote_matrix.shape
        out = self.imgOrigin.copy()
        # get x, y index of hough table
        # np.ravel Reduce the multidimensional array to 1 dimension
        ind_x = np.array(np.where(self.vote_matrix.ravel()>Hough_transform_threshold))[0]
        print("Number of votes of Line", self.vote_matrix.ravel()[ind_x])
        ind_y = ind_x.copy()
        thetas = ind_x % 180
        rhos = ind_y // 180
        for theta, rho in zip(thetas, rhos):    # each theta and rho
            t = np.pi / 180. * theta        # theta[radian] -> angle[degree]
            for x in range(self.x):
                if np.sin(t) != 0:
                    pass
                    y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
                    y = int(y)
                    if y >= self.y or y < 0:
                        continue
                    out[y, x] = [0,255,255]
            for y in range(self.y): # hough -> (x,y)
                if np.cos(t) != 0:
                    x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
                    x = int(x)
                    if x >= self.x or x < 0:
                        continue
                    out[y, x] = [0,0,255]
        return out.astype(np.uint8)

    # [Function name] Calculate
    # [Function Usage] Call all the above member functions in order and steps and return the detection results
    # [Parameter] None
    # [Return value] the picture with line being detected.
    # [Developer and date] Zhi DING 2020/12/11
    # [Change Record] None
    def Calculate(self):
        self.voting()
        out=self.inverse_hough()
        return out

if __name__ == '__main__':
    print("Please Input the name of the image which will be detected soon. For example, you can just input 'hw-coin.jpg' to detect this picture. Note: the picture which is going to be detected should be stored in the folder './source', and the pictrue which is processed will be saved in the folder './Output'. Wish you enjoy this program!")
    pictureName=input()
    img_gray = cv2.imread(Path+pictureName, cv2.IMREAD_GRAYSCALE)       # Read in grayscale image information
    img_RGB = cv2.imread(Path+pictureName)      # Read in colorful image information
    y, x = img_gray.shape[0:2]      # Import image size information
    print('[IN Canny Algorithm]')
    canny = Canny(Guassian_kernal_size, img_gray, HT_high_threshold, HT_low_threshold)  # Initialize the Canny class
    canny.canny_algorithm()     # Use Canny class functions for edge detection
    cv2.imwrite(Save_Path + pictureName + "_canny.jpg", canny.img)
    print('[IN Hough Transform Circle Detect Algorithm]')
    HoughCircle = Hough_Circle_Transform(canny.img, canny.angle, Hough_transform_step, Hough_transform_threshold)   # Initialize the Hough_Circle_Transform class
    circles = HoughCircle.Calculate()       # Use Hough_Circle_Transform class functions for Circle detection
    for circle in circles:      # Draw information about all circles
        cv2.circle(img_RGB, (math.ceil(circle[0]), math.ceil(circle[1])), math.ceil(circle[2]), (28, 36, 237), 2)
    cv2.imwrite(Save_Path + pictureName +"_hough_circle_result.jpg", img_RGB)
    print('[IN Hough Transform Line Detect Algorithm]')
    # cv2.imshow("result", canny.img)
    # HoughLine = Hough_Line_Transform(canny.img, img_RGB)
    HoughLine = Hough_Line_Transform(cv2.Canny(cv2.GaussianBlur(img_RGB,(23,23),0),75,150), img_RGB)       # Initialize the Hough_Line_Transform class
    HoughLineDetected=HoughLine.Calculate()       # Use Hough_Line_Transform class functions for Line detection
    cv2.imwrite(Save_Path + pictureName + "_hough_line_result.jpg", HoughLineDetected)
    print('[Contour/Line Extraction Finished]')