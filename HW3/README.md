## 项目目标

1.理解Harris Corner检测算法的实现原理，能自行实现相关功能

2.进一步熟练掌握OpenCV相关编程操作，熟悉OpenCV中有关Harris Corner检测的有关API

## 项目内容和原理

本实验主要内容为编写程序实现Harris Corner检测算法的相关功能。程序需要首先读入摄像头并回放视频。按一下空格键，则暂停回放，并将当前帧图像做一次Harris Corner检测，并将检测的结果叠加在原来图像上。实验代码应当含有如下内容：

1．需要自己写代码实现 Harris Corner 检测算法，不能直接调用OpenCV里面与 Harris角点检测相关的函数；

2.显示中间的处理结果及最终的检测结果，包括最大特征值图、最小特征值图、R图（可以考虑彩色展示）、原图上叠加检测结果等，并将中间结果都输出保存为图像文件。

角点检测(Corner Detection)也称为特征点检测，是图像处理和计算机视觉中用来获取图像局部特征点的一类方法，广泛应用于运动检测、图像匹配、视频跟踪、三维建模以及目标识别等领域中。

不同于HOG、LBP、Haar等基于区域(Region)的图像局部特征，Harris是基于角点的特征描述子，属于feature detector，主要用于图像特征点的匹配(match)，在SIFT算法中就有用到此类角点特征；而HOG、LBP、Haar等则是通过提取图像的局部纹理特征(feature extraction)，用于目标的检测和识别等领域。无论是HOG、Haar特征还是Harris角点都属于图像的局部特征，满足局部特征的一些特性。主要有以下几点：

-   可重复性(Repeatability)：同一个特征可以出现在不同的图像中，这些图像可以在不同的几何或光学环境下成像。也就是说，同一物体在不同的环境下成像(不同时间、不同角度、不同相机等)，能够检测到同样的特征。

-   独特性(Saliency)：特征在某一特定目标上表现为独特性，能够与场景中其他物体相区分，能够达到后续匹配或识别的目的。

-   局部性(Locality)；特征能刻画图像的局部特性，且对环境影响因子(光照、噪声等)鲁棒

-   紧致性和有效性(Compactness and efficiency)；特征能够有效地表达图像信息，而且在实际应用中运算要尽可能地快。

相比于考虑局部邻域范围的局部特征，全局特征则是从整个图像中抽取特征，较多地运用在图像检索领域，例如图像的颜色直方图。

除了以上几点通用的特性外，对于一些图像匹配、检测识别等任务，可能还需进一步考虑图像的局部不变特征。例如尺度不变性(Scale invariance)和旋转不变性(Rotation invariance)，当图像中的物体或目标发生旋转或者尺度发生变换，依然可以有效地检测或识别。此外，也会考虑局部特征对光照、阴影的不变性。

特征点在图像中一般有具体的坐标，并具有某些数学特征，如局部最大或最小灰度、以及某些梯度特征等。角点可以简单的认为是两条边的交点，比较严格的定义则是在邻域内具有两个主方向的特征点，也就是说在两个方向上灰度变化剧烈。如下图所示，在各个方向上移动小窗口，如果在所有方向上移动，窗口内灰度都发生变化，则认为是角点；如果任何方向都不变化，则是均匀区域；如果灰度只在一个方向上变化，则可能是图像边缘。

Harris角点检测的算法步骤归纳如下：

-   计算图像I(x,y)在X方向和Y方向的梯度

-   计算图像两个方向梯度的乘积I~2~x、I~2~y、IxIy

-   使用窗口高斯函数分别对I~2~x、I~2~y、IxIy进行高斯加权，生成矩阵M。

-   计算每个像素的Harris响应值R，并设定一阈值T，对小于阈值T的R置零。

-   在一个固定窗口大小的邻域内(5×5)进行非极大值抑制，局部极大值点即图像中的角点

Harris角点性质归纳如下：

-   参数α对角点检测的影响：增大α的值，将减小角点响应值R，减少被检测角点的数量；减小α的值，将增大角点响应值R，增加被检测角点的数量。

-   Harris角点检测对亮度和对比度的变化不敏感。

-   Harris角点检测具有旋转不变性，但不具备尺度不变性。如下图所示，在小尺度下的角点被放大后可能会被认为是图像边缘。

Harris角点具有灰度不变性和旋转不变性，但不具备尺度不变性，而尺度不变性对于图像的局部特征来说至关重要。将Harris角点检测算子和高斯尺度空间表示相结合，可有效解决这个问题。与Harris角点检测中的二阶矩表示类似，定义一个尺度自适应的二阶矩

式中，g(σI)表示尺度为σI的高斯卷积核，Lx(x,y,σD)和Ly(x,y,σD)表示对图像使用高斯函数g(σD)进行平滑后取微分的结果。σI称为积分尺度，是决定Harris角点当前尺度的变量，σD为微分尺度，是决定角点附近微分值变化的变量，通常σI应大于σD

算法流程如下：

-   确定尺度空间的一组取值σI=(σ0,σ1,σ2,...,σn)=(σ,kσ,k2σ,...,knσ),σD=sσI

-   对于给定的尺度空间值σD，进行角点响应值的计算和判断，并做非极大值抑制处理

-   在位置空间搜索候选角点后，还需在尺度空间上进行搜索，计算候选点的拉普拉斯响应值，并于给定阈值作比较

F(x,y,σn)=σ2n\|Lxx(x,y,σn)+Lyy(x,y,σn)\|≥threshold

-   将响应值F与邻近的两个尺度空间的拉普拉斯响应值进行比较，使其满足

F(x,y,σn)\>F(x,y,σl), l=n−1,n+1

-   这样既可确定在位置空间和尺度空间均满足条件的Harris角点。

## 操作方法与步骤

**Host Machine:** Windows 10 / Intel Core i5 1035G1 / 16GB RAM

**Python Version:** 3.8.5

**OpenCV Version:** 4.4.0

四、操作方法与实验步骤

### **功能点一**  读入摄像头，按下空格键后暂停视频并处理当前帧

OpenCV中提供了与摄像头信息处理相关的API函数，可利用OpenCV中相关函数接口实现摄像头信息的处理。对于空格键的按下后的判定，可参考第一次实验中对空格键的类似检测方法，本功能相关代码的核心部分如下：

cap **=** cv2**.**VideoCapture**(**0**)** \# 捕获摄像头信息

count**=**1

**while** **True:**

ret**,** frame **=** cap**.**read**()** \# 读取

cv2**.**namedWindow**(**\"Camara Capture\"**,** 0**)**

cv2**.**resizeWindow**(**\"Camara Capture\"**,** 800**,** 600**)**

cv2**.**imshow**(**\"Camara Capture\"**,** frame**)** \# 显示

**if** cv2**.**waitKey**(**100**)** **&** 0xff **==** **ord(**\' \'**):** \# 按空格键退出

imgname**=**\"currentFrame%s.jpg\"**%str(**count**)**

**print(**\"\[Frame %s\] Current Frame has been saved in current folder\"**%**count**)**

path **=** os**.**path**.**join**(**Save\_Path**,** imgname**)** \# 当前帧片保存路径

cv2**.**imwrite**(**path**,**frame**)**

count**+=**1

cv2**.**waitKey**(**0**)** \#break

cap**.**release**()**

cv2**.**destroyAllWindows**()**

### **功能点二** Harris Corner Detection算法

根据前文所述的Harris Corner Detection算法的主要步骤，可以逐步实现Harris Corner Detection算法如下所示：

#### Step1. 计算图像I(x,y)在X方向和Y方向的梯度

其相关数学原理如下公式所示，为了应用Harris Corner Detection算法，首先应当计算X方向及Y方向上的梯度。

在代码实现上，可以借助numpy库中的相关函数实现梯度的计算，相关核心代码如下：

dy**,** dx **=** np**.**gradient**(**img**)**

#### Step2.计算图像两个方向梯度的乘积I^2^x、I^2^y、IxIy

在上一步骤计算得到不同方向的梯度后，可以进一步计算两个方向上的梯度的乘积。其核心代码实现如下所示：

Ixx **=** dx**\*\***2

Ixy **=** dy**\***dx

Iyy **=** dy**\*\***2

#### Step3.使用窗口高斯函数分别对I^2^x、I^2^y、IxIy进行高斯加权，生成矩阵M

对于图片中的各个像素，可以进行加权运算，为后续步骤中Harris响应值的计算进行准备。本部分的核心代码实现如下所示：

height **=** img**.**shape**\[**0**\]**

width **=** img**.**shape**\[**1**\]**

offset **=** **int(**window\_size**/**2**)**

\#Loop through image and find our corners

**for** y **in** **range(**offset**,** height**-**offset**):**

**for** x **in** **range(**offset**,** width**-**offset**):**

\#Calculate sum of squares

windowIxx **=** Ixx**\[**y**-**offset**:**y**+**offset**+**1**,** x**-**offset**:**x**+**offset**+**1**\]**

windowIxy **=** Ixy**\[**y**-**offset**:**y**+**offset**+**1**,** x**-**offset**:**x**+**offset**+**1**\]**

windowIyy **=** Iyy**\[**y**-**offset**:**y**+**offset**+**1**,** x**-**offset**:**x**+**offset**+**1**\]**

Sxx **=** windowIxx**.sum()**

Sxy **=** windowIxy**.sum()**

Syy **=** windowIyy**.sum()**

#### Step4.计算每个像素的Harris响应值R，并设定一阈值T，根据响应值R检测图像的不同区域

根据Harris Corner Detection算法中响应值R的如下所示的计算公式，并结合前述步骤中已经计算出的各个变量的值，可以设计相关代码实现R的计算。在代码中阈值Threshold及系数α的值可以设为超参数，根据实际识别情况在一定范围内进行调整。

根据如下图所示的图像响应值R的不同大小与特征值大小的关系，可以设计出如下所示的代码用于检测图像中最大特征值区域、最小特征值区域，并在原图上叠加角点检测结果。同时下述代码还记录了各个像素对应的R值，用于后续R图绘制。

值得指出，在计算角点时，需要对一个固定窗口大小的邻域内(Windowsize×Windowsize)进行非极大值抑制，局部极大值点即图像中的角点。

与本步骤相关核心代码如下所示：

**for** y **in** **range(**offset**,** height**-**offset**):**

**for** x **in** **range(**offset**,** width**-**offset**):**

\#Find determinant and trace, use to get corner response

det **=** **(**Sxx **\*** Syy**)** **-** **(**Sxy**\*\***2**)**

trace **=** Sxx **+** Syy

r **=** det **-** k**\*(**trace**\*\***2**)**

\# cornerList.append(\[x, y, r\])

cornerList**\[**x**\]\[**y**\]=**r

**for** y **in** **range(**offset**,** height **-** offset**):**

**for** x **in** **range(**offset**,** width **-** offset**):**

**if** cornerList**\[**x**\]\[**y**\]** **\>** thresh**:**

**if** cornerList**\[**x**-**offset**:**x**+**offset**+**1**,**y**-**offset**:**y**+**offset**+**1**\].max()==**cornerList**\[**x**\]\[**y**\]:**

color\_img**.**itemset**((**y**,** x**,** 0**),** 0**)**

color\_img**.**itemset**((**y**,** x**,** 1**),** 0**)**

color\_img**.**itemset**((**y**,** x**,** 2**),** 255**)**

minEigenvalueImg**.**itemset**((**y**,** x**,** 0**),** 0**)**

minEigenvalueImg**.**itemset**((**y**,** x**,** 1**),** 0**)**

minEigenvalueImg**.**itemset**((**y**,** x**,** 2**),** 255**)**

**elif** cornerList**\[**x**\]\[**y**\]\<**0**:**

maxEigenvalueImg**.**itemset**((**y**,** x**,** 0**),** 0**)**

maxEigenvalueImg**.**itemset**((**y**,** x**,** 1**),** 0**)**

maxEigenvalueImg**.**itemset**((**y**,** x**,** 2**),** 255**)**

#### Step5. 根据前述步骤中获得的各个像素处的R值，绘制出图像的R值图

由于图像的R值是关于像素横纵坐标的二元函数，Matplotlib库封装了用于绘制类似场景的图片的函数，因而尝试使用Matplotlib库在三维空间内绘制此二元函数的值。核心代码如下所示：

fig **=** plt**.**figure**()**

ax **=** Axes3D**(**fig**)**

x **=** np**.**arange**(**0**,**data**.**shape**\[**0**\],**4**)**

y **=** np**.**arange**(**0**,**data**.**shape**\[**1**\],**3**)**

X**,** Y **=** np**.**meshgrid**(**x**,** y**)** \# \[important\] Create grid np.meshgrid(xnums,ynums)

Z**=**np**.**zeros**((**x**.**size**,**y**.**size**))**

**for** i **in** **range(**Z**.**shape**\[**0**\]):**

**for** j **in** **range(**Z**.**shape**\[**1**\]):**

Z**\[**i**\]\[**j**\]=**data**\[**x**\[**i**\],**y**\[**j**\]\]**

plt**.**xlabel**(**\'x\'**)**

plt**.**ylabel**(**\'y\'**)**

ax**.**plot\_surface**(**X**,** Y**,** Z**,** rstride**=**1**,** cstride**=**1**,** cmap**=**\'rainbow\'**)**

plt**.**savefig**(**Save\_Path**+**\'RValue%s.jpg\'**%str(**count**))**

## 结果与分析

本程序开始运行后，会检测程序同目录下是否有Output文件夹用于保存待输出的图片，若无此文件夹，则会新建这样的文件夹用于存储图片信息。随后开始采集摄像头信息，当按下空格键后，程序开始进行Harris Corner Detection运算，并将处理结果返回屏幕并输出；与此同时，Output文件夹下也会同时保存这些图片。

在一次处理结束后，只需要再次在摄像头窗口中按下空格键，即可再次开始捕获摄像头信息，随后继续按下空格键即可暂停并处理新的当前帧。如此周而复始直到用户关闭程序。

角点检测不仅重要，而且经常用到，它们也是计算机视觉中后续其他复杂检测的基础。通过本次实验，我初步熟悉了利用Harris Corner Detection实现角点检测的方法，此外在本次实验中我还进一步强化了对于非极大值抑制的认识。OpenCV中提供了基于Harris Corner Detection进行角点检测的API函数，通过本次实验后的尝试我注意到调用这些接口后的计算速度快于我所写的函数，一方面可能是因为我的算法函数还有很多优化空间，另一方面也可能是因为OpenCV底层基于C++语言实现，而我对Harris Corner Detection算法的实现基于Python，Python语言较慢的运行速度也使程序变慢。

Harris Corner Detection算法基本思想是使用一个固定窗口在图像上进行任意方向上的滑动，比较滑动前与滑动后两种情况，窗口中的像素灰度变化程度，如果存在任意方向上的滑动，都有着较大灰度变化，那么我们可以认为该窗口中存在角点。Harris在研究角点检测时也许对很多函数模型非常了解，对于创造出一个易于表示不同点特征值情况的函数表达式，从而极大地便利了角点检测。后人对Harris角点检测进行了很多进一步思考与改进。例如考虑到Harris角点具有光照不变性、旋转不变性、尺度不变性，但是严格意义上来说并不具备仿射不变性。Harris-Affine是一种新颖的检测仿射不变特征点的方法，可以处理明显的仿射变换，包括大尺度变化和明显的视角变化。Harris-Affine主要是依据了以下三个思路：①特征点周围的二阶矩的计算对区域进行的归一化，具有仿射不变性；②通过在尺度空间上归一化微分的局部极大值求解来精化对应尺度；③自适应仿射Harris检测器能够精确定位角点。诸如Harris-Affine的新的特征点检测方法也为计算机视觉的研究注入了新的活力。
