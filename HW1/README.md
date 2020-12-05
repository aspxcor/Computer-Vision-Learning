## Homework 1. Make a silent short video by using OpenCV
### Program description

> 1. Program to generate a new video meeting the following requirements. The generated video will be automatically saved in a video file with a specified file name after the program runs;
> 2. Content 1: At the beginning, it was a title, with lens switching, and showing related photos of Zhejiang University and personal photos. Show your student ID and name and other information.
> 3. Content 2: After the camera is switched, slowly draw a very simple children's drawing. The content of children's drawings is self-designed and written in the program. Cannot read a painted picture.
> 4. Other content: freely design by yourself
> 5. Content N: Programming an ending animation
> 6. Programming to achieve all lens switching effects;
> 7. There is a certain storyline, free design
> 8. Click the space to pause the video, and click again to continue.

### Note

**（一）内容一.片头部分**

根据实验要求，片头部分应当包括镜头切换，并显示相关照片

> （1）准备好片头所需要的相关图片素材，并放入统一的文件夹中，如source文件夹下
>
> （2）逐个加载各个有关元素相关的照片和个人大头照，关键代码框架如下（注：代码为实现所功能的核心部分，完整代码详见HW1.py文件，后文皆如此，不再赘述）
>
> **for** item **in** filelist**:**
>
> **if** item**.**endswith**(**\'.jpg\'**):**
>
> item **=** path **+** item
>
> img**.**append**(**cv2**.**imread**(**item**))**
>
> \# Get image size information
>
> **\[**height**,** width**,** pixels**\]** **=** img**\[-**1**\].**shape
>
> （3）考虑到不同图片尺寸可能存在差异，由于OpenCV在处理与输出尺寸不同的图片时可能会触发不可意料的错误（如无法正常输出视频等情况），因而需要逐个检查读入的图片尺寸是否符合预设视频输出尺寸信息，关键代码如下
>
> \# When the image size is not the same as the canvas size, use cv2.resize() to modify the image size to make it consistent with the canvas size
>
> **if** height **!=** size**\[**0**\]** **or** width **!=** size**\[**1**\]:**
>
> img**\[-**1**\]** **=** cv2**.**resize**(**img**\[-**1**\],** size**,** interpolation**=**cv2**.**INTER\_CUBIC**)**
>
> （4）根据实验要求，需要设计不同场景下的转场过渡效果，为此设计了渐入渐出的转场动画代码，核心原理是利用cv2.addWeighted方法合成出前后两个画面透明度不同的场景并输出，本部分核心代码如下
>
> **if** **len(**img**)** **\>** 1**:**
>
> **for** i **in** np**.**linspace**(**1**,** 0**,** 40**):**
>
> alpha **=** i
>
> beta **=** 1 **-** alpha
>
> output **=** cv2**.**addWeighted**(**img**\[-**2**\],** alpha**,** img**\[-**1**\],** beta**,** 0**)**
>
> \# Add subtitle description file
>
> cv2**.**putText**(**output**,** \'\'\'Content I: Pictures\'\'\'**,(**100**,** 100**),** font**,** 1**,** **(**55**,** 255**,** 155**),** 1**)**
>
> video**.**write**(**output**)**
>
> cv2**.**imshow**(**\'HW1\'**,** output**)**
>
> （5）根据实验要求，需要设计实现按下空格后暂停播放，再次按下空格后恢复播放的效果。查阅OpenCV相关文档后了解到可借助cv2.waitKey方法实现，核心代码如下
>
> \# The following four lines of code are used to implement the eighth function in the experiment description, that is, click on the space to pause the video, and click again to continue.
>
> **if** cv2**.**waitKey**(**1**)** **==** 27**:**
>
> **break**

**if** cv2**.**waitKey**(**1**)** **&** 0xFF **==** **ord(**\' \'**):**

cv2**.**waitKey**(**0**)**

> （6）根据实验要求，设计本部分内容结束后切换到下一部分内容时的过渡动画，过渡动画相关代码设计与前文类似，需新建黑色画布用于下一场景制图及两场景间过渡，下一画布新建的代码如下
>
> \# After finishing loading Zhejiang University and personal information pictures, the empty canvas will be loaded below, and the transition will be made to the empty canvas for the drawing of the children's drawings in the second part.
>
> imag **=** np**.**zeros**((**1080**,** 1920**,** 3**),** np**.**uint8**)**

（7）完成本段程序后返回新建的画布，用于后续函数绘制儿童画使用。

**（二）内容二.儿童画部分**

根据实验要求，儿童画部分应当缓慢画一幅很简单的儿童画，儿童画内容自行设计，写死在程序里面，注意不能读入一个画好的图片。

结合实验内容，应当注意到本部分应当使用诸如cv2.line()、cv2.circle()、cv2.rectangle()之类的方式绘制儿童画。需要考虑的是如何缓慢绘制，此处我使用了for循环来实现缓慢绘制的效果。一段缓慢绘制直线的程序的核心部分如下所示，图画中每一部分均可参照此方法逐步完成绘制。

**for** i **in** **range(**480**):**

cv2**.**line**(**img**,** **(**0**,** 800**),** **(**i**\***4**,** 800**),** **(**240**,** 240**,** 240**),** 10**)**

cv2**.**imshow**(**\'HW1\'**,** img**)**

video**.**write**(**img**)**

**（三）内容三.自由设计部分------风光片节选**

根据实验要求，本部分的内容和实现方式可自行设计。为全面把握和了解OpenCV的各项功能的使用，本部分中我设计了读取一段风光片的片段，并将之输出为视频的功能。

> （1）准备好片头所需要的相关视频素材，并放入统一的文件夹中，如source文件夹下
>
> （2）利用cv2.VideoCapture()及其相关函数实现视频逐帧读入，逐帧读入后检测当前帧内是否有信息，如有，则输出到屏幕上并写入之后将存盘的视频中；如没有信息（也即已经读取完全部帧），则需要将视频的最后一帧保存并返回，用于实现与之后部分的过渡动画。本部分核心代码如下。
>
> **def** story**(**video**,**preImg**,**name**):**
>
> videoCapture **=** cv2**.**VideoCapture**(**path **+** name**)**
>
> success**,** frame **=** videoCapture**.**read**()** \# When reading the video, videoCapture.read() returns two values, which are respectively the Boolean value success used to mark whether the current frame has been read successfully (when there is no new frame that can be read, it returns false, otherwise it returns True), frame is the frame that is read out
>
> \# Loop in the given video until the video is completely read
>
> **while** success**:**
>
> cv2**.**imshow**(**\'HW1\'**,** frame**)**
>
> **if** cv2**.**waitKey**(**1**)** **==** 27**:**
>
> **break**
>
> **if** cv2**.**waitKey**(**1**)** **&** 0xFF **==** **ord(**\' \'**):**
>
> cv2**.**waitKey**(**0**)**
>
> video**.**write**(**frame**)**
>
> success**,** tempFrame **=** videoCapture**.**read**()**
>
> \# tempFrame saves the last frame of the previously read video picture at all times, and the final value of tempFrame is the last frame of the video to be read, used for subsequent scene transitions
>
> **if** success**:**
>
> frame**=**tempFrame
>
> **return** frame

**（四）内容四.片尾动画部分------熊猫行走**

根据实验要求，本部分需编程实现一个片尾动画，具体内容没有明确要求。本部分中设计了一个熊猫行走的动画效果，并在屏幕上不断显示一系列计算机科学相关名言。实现思路为令熊猫原地不动，使背景图片反向运动，使人在视觉上有了一种熊猫在向前运动的感觉。

> （1）准备好片头所需要的相关视频素材，并统一放入Animation文件夹下。
>
> （2）利用cv2.imread ()及其相关函数实现背景图片信息的载入，背景图片信息分为两类------一类是初始加载的背景，一类是随着熊猫的运动而不断载入的新的背景图片。新的背景图片不断载入以实现熊猫仿佛在运动的视觉感受。两部分函数分别封装为getBackground(name)及getImage(name)，其中getBackground(name)函数细节如下（注：getImage(name)函数细节类似，此处略去）
>
> **def** getBackground**(**name**):**
>
> **global** speed
>
> a**=**cv2**.**imread**(**r\'{}\'**.format(**name**))**
>
> lenx**=len(**a**\[**1**,:\])**
>
> leny**=len(**a**\[:,**1**\])**
>
> modx**=**lenx**%**speed
>
> mody**=**leny**%**speed
>
> **if(**modx**!=**0 **or** mody**!=**0**):**
>
> **if** modx**!=**0**:**
>
> lenx**+=**modx
>
> **if** mody**!=**0**:**
>
> leny**+=**mody
>
> a**=**cv2**.**resize**(**a**,(**lenx**,**leny**),**interpolation**=**cv2**.**INTER\_AREA**)**
>
> **return** a
>
> （3）在动画开始生成前逐个加载背景图片，加载完成初始背景后实现熊猫动画部分与前一部分的过渡动画。
>
> （4）借助random库的帮助，在while循环中不断加载不同背景图片信息实现熊猫运动的动画效果，直到time库计时达到预设值时停止。核心代码如下。
>
> flag **=** 0
>
> flag2 **=** 0
>
> flag3 **=** 1
>
> old\_i **=** population**\[**0**\].**copy**()**
>
> \# Timing starts, timing is achieved through the time library
>
> tBegin **=** time**.**time**()**
>
> \# Continue to load and output animation information until the timing duration reaches the preset value specified by the hyperparameter ANIMATION\_DURATION
>
> **while** **(**time**.**time**()** **-** tBegin **\<** ANIMATION\_DURATION**):**
>
> count **=** 0
>
> **if** **(**flag **==** 1**):**
>
> count **=** **-**1
>
> random**.**shuffle**(**population**)**
>
> \# Continue to load and output animation information until the timing duration reaches the preset value specified by the hyperparameter ANIMATION\_DURATION
>
> **while** **(**count **\<** **len(**population**)** **-** 1 **and** time**.**time**()** **-** tBegin **\<** ANIMATION\_DURATION**):**
>
> x **=** 0
>
> recount **=** count **+** 1
>
> j **=** population**\[**recount**\].**copy**()**
>
> \# Continue to load and output animation information until the timing duration reaches the preset value specified by the hyperparameter ANIMATION\_DURATION
>
> **while** **(**x **!=** **len(**j**\[**1**,** **:\])** **and** time**.**time**()** **-** tBegin **\<** ANIMATION\_DURATION**):**
>
> x **+=** speed
>
> a **=** 0
>
> b **=** speed
>
> d **=** **len(**i**\[**1**,** **:\])**
>
> c **=** d **-** b
>
> copy1 **=** j**\[:,** a**:**b**\].**copy**()**
>
> copy2 **=** j**\[:,** b**:**d**\].**copy**()**
>
> copy3 **=** i**\[:,** b**:**d**\].**copy**()**
>
> new\_var **=** 0
>
> **while** **(**new\_var **\<** 1**):**
>
> ranger **=** **range(**0**,** 994**)**
>
> i**\[**987**:**990**,** **:\]** **=** 0
>
> range1 **=** random**.**choice**(**ranger**)**
>
> i**\[**992**:**995**,** range1**:**range1 **+** 6**\]** **=** 45
>
> new\_var **+=** 1
>
> key **=** cv2**.**waitKey**(**10**)**
>
> i**\[:,** a**:**c**\]** **=** copy3
>
> i**\[:,** c**:\]** **=** copy1
>
> j**\[:,** a**:**c**\]** **=** copy2
>
> \# Different output according to different flag values to realize the dynamic effect of panda walking
>
> **if** flag2 **==** 0**:**
>
> \# Defines the coordinates of the panda in the animation
>
> old\_i**\[**920**:**1020**,** 100**:**200**\]** **=** i**\[**920**:**1020**,** 100**:**200**\].**copy**()**
>
> flag2 **=** 1
>
> **if** flag2 **==** 1 **and** flag3 **==** 1**:**
>
> i**\[**690**:**990**,** 100**:**400**\]** **=** panda**.**copy**()**
>
> **if** key **==** **ord(**\'w\'**):**
>
> flag3 **=** 0
>
> **if** flag3 **==** 0**:**
>
> i**\[**920**:**1020**,** 100**:**200**\]** **=** old\_i**\[**920**:**1020**,** 100**:**200**\]**
>
> pixcounter **=** 0
>
> **while** **(**pixcounter **\<** 100**):**
>
> a1 **=** 920 **-** pixcounter
>
> b1 **=** 1020 **-** pixcounter
>
> delay **=** 0
>
> **while** **(**delay **!=** 1000**):**
>
> delay **+=** 1
>
> **pass**
>
> pixcounter **+=** 1
>
> q1 **=** i**\[**a1**:**b1**,** 100**:**200**\].**copy**()**
>
> i**\[**a1**:**b1**,** 100**:**200**\]** **=** q1
>
> i**\[**320 **-** pixcounter**:**420 **-** pixcounter**,** 100**:**200**\]** **=** panda**.**copy**()**
>
> count **+=** 1
>
> flag **=** 1
>
> （5）在动画页面空白部分显示与计算机科学相关的名人名言，核心代码如下
>
> celebrities**=\[**
>
> \'\'\'The Internet? Is that thing still around? \--Homer Simpson\'\'\'**,**
>
> \'\'\'Talk is cheap. Show me the code. \--Linus Torvalds\'\'\'**,**
>
> \'\'\'Computers are useless. They can only give you answers. \--Pablo Picasso\'\'\'**,**
>
> \'\'\'Good code is its own best documentation. \--Steve McConnell\'\'\'**,**
>
> \'\'\'Software is like sex: It\'s better when it\'s free. \--Linus Torvalds\'\'\'
>
> **\]** \# Some interesting celebrities in computer science
>
> \# OutPut the celebrities
>
> **try:**
>
> cv2**.**putText**(**i**,** celebrities**\[int((**time**.**time**()** **-** tBegin**)//**2**)\],** **(**500**,** 900**),** font**,** 1**,(**0**,** 0**,** 255**),** 2**)**
>
> **except:**
>
> **pass**

**（五）**`main`**函数**

实现完各部分功能后，定义main函数如下所示。注意在程序开始前需要初始化播放窗口并开始输出视频的相关设置，程序运行结束后保存视频并输出，随后关闭视频播放窗口。

**if** \_\_name\_\_ **==** \'\_\_main\_\_\' **:**

\# Define the video name and format information of the final output of the program

video **=** cv2**.**VideoWriter**(**\"HW1.mp4\"**,** cv2**.**VideoWriter\_fourcc**(**\'M\'**,**\'P\'**,**\'4\'**,**\'V\'**),** fps**,** size**)**

\# Define the name of the video playback window

cv2**.**namedWindow**(**\'HW1\'**,** cv2**.**WINDOW\_NORMAL**)**

\# Load the title picture, including related pictures and personal photos of Zhejiang University

img**=**loadCredits**()**

\# Load children\'s drawing fragments and draw children\'s drawings

pictureImg**=**drawPicture**(**video**,**img**)**

\# Load the short story clips and read some excerpts from the promotional video of Zhejiang University

lastFrameOfStory**=**story**(**video**,**pictureImg**,**\'Input.mp4\'**)**

\# Load the animation of the ending panda walking

pandaAnimation**(**lastFrameOfStory**)**

\# Stop video output, the currently loaded video will be saved and output

video**.**release**()**

\# Close the playback control window

cv2**.**destroyAllWindows**()**
