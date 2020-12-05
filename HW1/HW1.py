# # Homework 1 Name: Make a silent short video by using OpenCV
# # Program description:
    # 1. Program to generate a new video meeting the following requirements. The generated video will be automatically saved in a video file with a specified file name after the program runs;
    # 2. Content 1: At the beginning, it was a title, with lens switching, and showing related photos of Zhejiang University and personal photos. Show your student ID and name and other information.
    # 3. Content 2: After the camera is switched, slowly draw a very simple children's drawing. The content of children's drawings is self-designed and written in the program. Cannot read a painted picture.
    # 4. Other content: freely design by yourself.
    # 5. Content N: Programming an ending animation
    # 6. Programming to achieve all lens switching effects;
    # 7. There is a certain storyline, free design
    # 8. Click the space to pause the video, and click again to continue.
# # File Name: HW1.py
# # Last Modified: 2020/11/29

import os
import cv2
import numpy as np
import time
import random

# Some global variables and basic hyperparameter information are defined here
path = './source/'      # Source file directory
filelist = os.listdir(path)
fps = 24                # The video's sampling rate is 24 frames per second
size = (1920, 1080)     # Target output size of this video
font = cv2.FONT_HERSHEY_SIMPLEX     # Subtitle font in this video
speed=4                 # Canvas movement speed in panda animation
img=[]                  # Used to load the list of opening photos, initially empty
ANIMATION_DURATION=10   # Panda animation duration

# [Function name] loadCredits
# [Function Usage] This function is used to implement the first function in the experimental requirements, that is, the content related to the title, there is a lens switch in the title part, and the related photos of Zhejiang University elements and personal photos are displayed. Information such as your student ID and name should be displayed in the title.
# [Parameter] None
# [Return value] Returns the `imag` which is the image on which text and picture are to be drawn in the next scene.
# [Developer and date] Anonymous
# [Change Record] None
def loadCredits():
    # Traverse all the files in the source folder and select all picture files with the suffix .jpg
    for item in filelist:
        if item.endswith('.jpg'):
            item = path + item
            img.append(cv2.imread(item))
            # Get image size information
            [height, width, pixels] = img[-1].shape
            # When the image size is not the same as the canvas size, use cv2.resize() to modify the image size to make it consistent with the canvas size
            if height != size[0] or width != size[1]:
                img[-1] = cv2.resize(img[-1], size, interpolation=cv2.INTER_CUBIC)
                # When more than one picture appears, a transition effect between different pictures needs to be performed. The following code implements a slow transition effect
            if len(img) > 1:
                for i in np.linspace(1, 0, 40):
                    alpha = i
                    beta = 1 - alpha
                    output = cv2.addWeighted(img[-2], alpha, img[-1], beta, 0)
                    # Add subtitle description file
                    cv2.putText(output, '''Content I: Pictures''',(100, 100), font, 1, (55, 255, 155), 1)
                    video.write(output)
                    cv2.imshow('HW1', output)
                    # time.sleep(0.02)
                    # The following four lines of code are used to implement the eighth function in the experiment description, that is, click on the space to pause the video, and click again to continue. I will not repeat the description when encountering similar functions later.
                    if cv2.waitKey(1) == 27:
                        break
                    if cv2.waitKey(1) & 0xFF == ord(' '):
                        cv2.waitKey(0)
    # After finishing loading Zhejiang University and personal information pictures, the empty canvas will be loaded below, and the transition will be made to the empty canvas for the drawing of the children’s drawings in the second part.
    imag = np.zeros((1080, 1920, 3), np.uint8)
    # When more than one picture appears, a transition effect between different pictures needs to be performed. The following code implements a slow transition effect
    for i in np.linspace(1, 0, 40):
        alpha = i
        beta = 1 - alpha
        output = cv2.addWeighted(img[-1], alpha, imag, beta, 0)
        video.write(output)
        cv2.imshow('HW1', output)
        # time.sleep(0.02)
        if cv2.waitKey(1) == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.waitKey(0)
    return imag

# [Function name] drawPicture
# [Function Usage] After the scene is switched, the function of slowly drawing a children's picture. The content of children's drawings is composed of two parts: the ground and the sun. The content of children's drawings is defined in the program body instead of reading a drawn picture.
# [Parameter] video: The VideoWirter which is initialized by the function `cv2.VideoWriter`;
#             img: It is the image on which text and picture are to be drawn.
# [Return value] Returns the image on which text and picture are to be drawn, the image is used to create the opening transition animation in the next scene
# [Developer and date] Anonymous
# [Change Record] None
def drawPicture(video,img):
    # Add subtitle description file
    cv2.putText(img, '''Content II: Children's Drawing''', (100, 100),font, 1, (55, 255, 155),1)
    # This loop is used to slowly draw a flat ground
    for i in range(480):
        cv2.line(img, (0, 800), (i*4, 800), (240, 240, 240), 10)
        cv2.imshow('HW1', img)
        video.write(img)
        # The following four lines of code are used to implement the eighth function in the experiment description, that is, click on the space to pause the video, and click again to continue. I will not repeat the description when encountering similar functions later.
        if cv2.waitKey(1) == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.waitKey(0)
    # This loop is used to slowly draw the circular part of the sun
    for i in range(75):
        cv2.circle(img, (300, 300), i*2, (0, 0, 255), 2)
        cv2.imshow('HW1', img)
        video.write(img)
        if cv2.waitKey(1) == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.waitKey(0)
    # The following four loops are used to slowly draw the rays of the sun
    for i in range(25):
        cv2.line(img, (450, 300), (450+i*2, 300), (0, 240, 255), 5)
        cv2.imshow('HW1', img)
        video.write(img)
        if cv2.waitKey(1) == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.waitKey(0)
    for i in range(25):
        cv2.line(img, (150, 300), (150-i*2, 300), (0, 240, 255), 5)
        cv2.imshow('HW1', img)
        video.write(img)
        if cv2.waitKey(1) == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.waitKey(0)
    for i in range(25):
        cv2.line(img, (300, 450), (300, 450+i*2), (0, 240, 255), 5)
        cv2.imshow('HW1', img)
        video.write(img)
        if cv2.waitKey(1) == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.waitKey(0)
    for i in range(25):
        cv2.line(img, (300, 150), (300, 150-i*2), (0, 240, 255), 5)
        cv2.imshow('HW1', img)
        video.write(img)
        if cv2.waitKey(1) == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.waitKey(0)
    return img

# [Function name] story
# [Function Usage] This function implements the third part of the job description, which is to realize the open design content. In this clip, I designed related functions by reading in a video and exporting it again.
# [Parameter] video: The VideoWirter which is initialized by the function `cv2.VideoWriter`;
#             preImg: It is the image on which text and picture were to be drawn in the past function;
#             name: It is the name of the video which is going to be read in.
# [Return value] Returns the last frame of the video which is read in.
# [Developer and date] Anonymous
# [Change Record] None
def story(video,preImg,name):
    videoCapture = cv2.VideoCapture(path + name)
    success, frame = videoCapture.read()       # When reading the video, videoCapture.read() returns two values, which are respectively the Boolean value success used to mark whether the current frame has been read successfully (when there is no new frame that can be read, it returns false, otherwise it returns True), frame is the frame that is read out
    # The transition part of the fade-in and fade-out effect between the canvas drawn by the child in the previous scene and the canvas of the video clip
    for i in np.linspace(1, 0, 40):
        alpha = i
        beta = 1 - alpha
        output = cv2.addWeighted(preImg, alpha, frame, beta, 0)
        video.write(output)
        cv2.imshow('HW1', output)
        # time.sleep(0.02)
        if cv2.waitKey(1) == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.waitKey(0)
    # Loop in the given video until the video is completely read
    while success:
        cv2.putText(frame, '''Content III: A Story''', (100, 100),font, 1, (55, 255, 155), 1)
        cv2.imshow('HW1', frame)
        if cv2.waitKey(1) == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.waitKey(0)
        video.write(frame)
        success, tempFrame = videoCapture.read()
        # tempFrame saves the last frame of the previously read video picture at all times, and the final value of tempFrame is the last frame of the video to be read, used for subsequent scene transitions
        if success:
            frame=tempFrame
    return frame

# [Function name] getBackground
# [Function Usage] This function will be called in the pandaAnimation function to load the canvas of the background image in the panda animation scene.
# [Parameter] name:The name of the image to be loaded, marked with a relative path or an absolute path
# [Return value] Return to the canvas with the background image loaded
# [Developer and date] Anonymous
# [Change Record] None
def getBackground(name):
    global speed
    a=cv2.imread(r'{}'.format(name))
    lenx=len(a[1,:])
    leny=len(a[:,1])
    modx=lenx%speed
    mody=leny%speed
    if(modx!=0 or mody!=0):
        if modx!=0:
            lenx+=modx
        if mody!=0:
            leny+=mody
        a=cv2.resize(a,(lenx,leny),interpolation=cv2.INTER_AREA)
    return a

# [Function name] getImage
# [Function Usage] This function will be called in the pandaAnimation function to load the canvas of the other image in the panda animation scene.
# [Parameter] name:The name of the image to be loaded, marked with a relative path or an absolute path
# [Return value] Return to the canvas with the other image loaded
# [Developer and date] Anonymous
# [Change Record] None
def getImage(name):
    a=cv2.imread(r'{}'.format(name))
    global imagex, imagey
    a=cv2.resize(a,(imagex,imagey),interpolation=cv2.INTER_AREA)
    return a

# [Function name] pandaAnimation
# [Function Usage] This function is used to implement the last part of the experiment description, which is to design an ending animation by yourself. This program implements an animation of a panda walking on the ground, which is realized by the program instead of reading the existing animation.
# [Parameter] preFrame:The canvas in the previous scene, used to implement scene transition in this function
# [Return value] None
# [Developer and date] Anonymous
# [Change Record] None
def pandaAnimation(preFrame):
    # Load each picture information, initialize the canvas
    population=[]       # The list named population contains the background image and several images that will be loaded in the animation
    celebrities=[
        '''The Internet? Is that thing still around?  --Homer Simpson''',
        '''Talk is cheap. Show me the code.  --Linus Torvalds''',
        '''Computers are useless. They can only give you answers.  --Pablo Picasso''',
        '''Good code is its own best documentation.  --Steve McConnell''',
        '''Software is like sex: It's better when it's free.  --Linus Torvalds'''
    ]   # Some interesting celebrities in computer science
    population.append(getBackground('./Animation/background.jpg'))
    global imagex, imagey
    imagex = len(population[0][1, :])
    imagey = len(population[0][:, 1])
    for i in range(1,4):
        population.append(getImage('./Animation/' +str(i) + '.jpg'))
    panda = cv2.imread(r'./Animation/panda.jpg')
    i = population[0].copy()
    # Transition between this canvas and the previous canvas to achieve transition animation effect
    for p in np.linspace(1, 0, 40):
        alpha = p
        beta = 1 - alpha
        tempFrame = cv2.addWeighted(preFrame, alpha, i, beta, 0)
        video.write(tempFrame)
        cv2.imshow('HW1', tempFrame)
        # time.sleep(0.02)
        if cv2.waitKey(1) == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.waitKey(0)
    flag = 0
    flag2 = 0
    flag3 = 1
    old_i = population[0].copy()
    # Timing starts, timing is achieved through the time library
    tBegin = time.time()
    # Continue to load and output animation information until the timing duration reaches the preset value specified by the hyperparameter ANIMATION_DURATION
    while (time.time() - tBegin < ANIMATION_DURATION):
        count = 0
        if (flag == 1):
            count = -1
        random.shuffle(population)
        # Continue to load and output animation information until the timing duration reaches the preset value specified by the hyperparameter ANIMATION_DURATION
        while (count < len(population) - 1 and time.time() - tBegin < ANIMATION_DURATION):
            x = 0
            recount = count + 1
            j = population[recount].copy()
            # Continue to load and output animation information until the timing duration reaches the preset value specified by the hyperparameter ANIMATION_DURATION
            while (x != len(j[1, :]) and time.time() - tBegin < ANIMATION_DURATION):
                print(time.time() - tBegin)     # Output current timing value for debugging
                if cv2.waitKey(1) == 27:
                    break
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    cv2.waitKey(0)
                x += speed
                a = 0
                b = speed
                d = len(i[1, :])
                c = d - b
                copy1 = j[:, a:b].copy()
                copy2 = j[:, b:d].copy()
                copy3 = i[:, b:d].copy()
                new_var = 0
                while (new_var < 1):
                    ranger = range(0, 994)
                    i[987:990, :] = 0
                    range1 = random.choice(ranger)
                    i[992:995, range1:range1 + 6] = 45
                    new_var += 1
                cv2.putText(i, '''Content IV: Ending Animation: Panda Walking''', (100, 100), font, 1, (0, 0, 0),1)
                # OutPut the celebrities
                try:
                    cv2.putText(i, celebrities[int((time.time() - tBegin)//2)], (500, 900), font, 1,(0, 0, 255), 2)
                except:
                    pass
                cv2.imshow('HW1', i)
                video.write(i)
                key = cv2.waitKey(10)
                i[:, a:c] = copy3
                i[:, c:] = copy1
                j[:, a:c] = copy2
                # Different output according to different flag values to realize the dynamic effect of panda walking
                if flag2 == 0:
                    # Defines the coordinates of the panda in the animation
                    old_i[920:1020, 100:200] = i[920:1020, 100:200].copy()
                    flag2 = 1
                if flag2 == 1 and flag3 == 1:
                    i[690:990, 100:400] = panda.copy()
                if key == ord('w'):
                    flag3 = 0
                if flag3 == 0:
                    i[920:1020, 100:200] = old_i[920:1020, 100:200]
                    pixcounter = 0
                    while (pixcounter < 100):
                        a1 = 920 - pixcounter
                        b1 = 1020 - pixcounter
                        delay = 0
                        while (delay != 1000):
                            delay += 1
                            pass
                        pixcounter += 1
                        q1 = i[a1:b1, 100:200].copy()
                        i[a1:b1, 100:200] = q1
                        i[320 - pixcounter:420 - pixcounter, 100:200] = panda.copy()
            count += 1
            flag = 1

# Main program entry
if __name__ == '__main__' :
    # Define the video name and format information of the final output of the program
    video = cv2.VideoWriter("HW1.mp4", cv2.VideoWriter_fourcc('M','P','4','V'), fps, size)
    # Define the name of the video playback window
    cv2.namedWindow('HW1', cv2.WINDOW_AUTOSIZE)
    # Load the title picture, including related pictures and personal photos of Zhejiang University
    img=loadCredits()
    # Load children's drawing fragments and draw children's drawings
    pictureImg=drawPicture(video,img)
    # Load the short story clips and read some excerpts from the promotional video of Zhejiang University
    lastFrameOfStory=story(video,pictureImg,'Input.mp4')
    # Load the animation of the ending panda walking
    pandaAnimation(lastFrameOfStory)
    # Stop video output, the currently loaded video will be saved and output as set
    video.release()
    # Close the playback control window
    cv2.destroyAllWindows()