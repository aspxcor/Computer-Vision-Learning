一、实验目的和要求

1.通过制作一个简单的视频来掌握使用OpenCV进行数字图像处理的基本技能

2.掌握OpenCV进行曲线检测的方法

二、实验内容和原理

本实验主要内容为使用OpenCV库对输入的一张彩色图像，检测其中的圆形与直线，并将检测结果显示在原图上。实验代码应当含有如下内容。

1\. 检测算法的核心功能需要自己 写代码实现，不能调用 OpenCV或其他SDK里与圆形 直线检测相关的函数；如果要用到边缘检测，这个可以调用OpenCV函数。

2.在原图上显示最终的检测结果；

3.单独显示一些关键的中间结果

4.必须对指定的三张测试图像（coin、seal、highway）调试结果。此外，自己还可以自愿加一些测试图像。

本实验中拟采用Canny算法实现边缘检测。图象边缘检测必须满足两个条件，其一是能有效地抑制噪声；其二是必须尽量精确确定边缘的位置。根据对信噪比与定位乘积进行测度，得到最优化逼近算子。这就是Canny边缘检测算子，这类似与Marr(LoG)边缘检测方法，也属于先平滑后求导数的方法。

本实验中拟采用Hough变换实现圆与直线的检测。霍夫变换是图像处理中的一种特征提取技术，它通过一种投票算法检测具有特定形状的物体。Hough变换是图像处理中从图像中识别几何形状的基本方法之一。Hough变换的基本原理在于利用点与线的对偶性，将原始图像空间的给定的曲线通过曲线表达形式变为参数空间的一个点。这样就把原始图像中给定曲线的检测问题转化为寻找参数空间中的峰值问题。也即把检测整体特性转化为检测局部特性。比如直线、椭圆、圆、弧线等。

Hough变换也可以检测任意的已知表达形式的曲线，关键是看其参数空间的选择，参数空间的选择可以根据它的表达形式而定。因而Hough变换可以推广至对圆和椭圆的检测，具体的来说，若要检测已知半径的圆，可以选择与原图像空间同样的空间作为参数空间。那么圆图像空间中的一个圆对应了参数空间中的一个点，参数空间中的一个点对应了图像空间中的一个圆，圆图像空间中在同一个圆上的点，它们的参数相同即a，b相同，那么它们在参数空间中的对应的圆就会过同一个点(a，b)，所以，将原图像空间中的所有点变换到参数空间后，根据参数空间中点的聚集程度就可以判断出图像空间中有没有近似于圆的图形。如果有的话，这个参数就是圆的参数。对于未知半径的圆的检测，在圆的半径未知的情况下，可以看作是有三个参数的圆的检测，中心和半径。这个时候原理仍然相同，只是参数空间的维数升高，计算量增大。图像空间中的任意一个点都对应了参数空间中的一簇圆曲线，其实是一个圆锥型。参数空间中的任意一个点对应了图像空间中的一个圆。对于椭圆的检测，由于椭圆有5个自由参数，所以它的参数空间是5维的，因此其计算量非常大，所以提出了许多的改进算法。

三、主要仪器设备

**Host Machine:** Windows 10 / Intel Core i5 1035G1 / 16GB RAM

**Python Version:** 3.8.5

**OpenCV Version:** 4.4.0

四、操作方法与实验步骤

**（一）基于Canny算法的边缘检测**

基于Canny算法的原理，可以复现出基于Canny算法的边缘检测。算法实现过程中的一些核心要点可概括为如下几点，下面的报告中仅展示核心算法思想及部分函数接口说明，全部代码详见源文件。

（1）Canny算法实现边缘检测的核心思想

对于Canny边缘检测算法可按照如下思路进行设计：

> Step1.用高斯滤波器平滑图象；
>
> Step2.用一阶偏导的有限差分来计算梯度的幅值和方向；
>
> Step3.对梯度幅值进行非极大值抑制；
>
> Step4.用双阈值算法检测和连接边缘。

（2）Canny边缘检测算法的接口设计

在本次实验中，我设计了Canny类以实现边缘检测，这个类的几个函数原型及各函数功能如下：

**class** **Canny:**

**def** \_\_init\_\_**(**self**,** Guassian\_kernal\_size**,** img**,** HT\_high\_threshold**,** HT\_low\_threshold**):**

\'\'\'

:Usage: 初始化\_\_init\_\_类

:param Guassian\_kernal\_size: 高斯滤波器尺寸

:param img: 输入的图片，在算法过程中改变

:param HT\_high\_threshold: 滞后阈值法中的高阈值

:param HT\_low\_threshold: 滞后阈值法中的低阈值

\'\'\'

**def** Get\_gradient\_img**(**self**):**

\'\'\'

:Usage: 计算梯度图和梯度方向矩阵。

:return: 生成的梯度图

\'\'\'

**def** Non\_maximum\_suppression**(**self**):**

\'\'\'

:Usage: 对生成的梯度图进行非极大化抑制，将tan值大小与正负结合，确定离散中梯度的方向

:return: 生成的非极大化抑制结果图

\'\'\'

**def** Hysteresis\_thresholding**(**self**):**

\'\'\'

:Usage: 对生成的非极大化抑制结果图进行滞后阈值法，用强边延伸弱边，这里的延伸方向为梯度的垂直方向，将比低阈值大比高阈值小的点置为高阈值大小，方向在离散点上的确定与非极大化抑制相似

:return: 滞后阈值法结果图

\'\'\'

**def** canny\_algorithm**(**self**):**

\'\'\'

:Usage: 按照顺序和步骤调用以上所有成员函数。

:return: Canny 算法的结果

\'\'\'

**（二）基于Hough算法的圆的检测**

基于Hough算法的原理，可以复现出基于Hough算法的圆的检测。算法实现过程中的一些核心要点可概括为如下几点，下面的报告中仅展示核心算法思想及部分函数接口说明，全部代码详见源文件。

（1）Hough变换实现圆检测算法的核心思想

对于圆弧检测的Hough算法可按照如下思路进行设计：

> Step1.量化与待检测图形有关的参数空间到合适精度
>
> Step2.初始化所有累加器为0
>
> Step3.计算图像空间中边缘点的梯度幅度和角度
>
> Step4.若边缘点参数坐标满足，则该参数坐标对应的累加器加1
>
> Step5.拥有最大值的累加器所在的坐标即为图像空间中的圆心之所在

Step6.得到圆心坐标之后，我们可以求解半径r并利用cv2.circle()函数绘图

对于Hough算法实现圆的检测的核心算法伪代码如下（全部代码详见源码）：

![](media/image2.png){width="3.2485170603674542in" height="1.2097900262467192in"}

（2）Hough变换实现圆检测算法的接口设计

在本次实验中，我设计了Hough\_Circle\_Transform类以实现圆的检测，这个类的几个函数原型及各函数功能如下：

**class** **Hough\_Circle\_Transform:**

**def** \_\_init\_\_**(**self**,** img**,** angle**,** step**=**5**,** threshold**=**135**):**

\'\'\'

:Usage: 初始化Hough\_Circle\_Transform类

:param img: 输入的图像

:param angle: 输入的梯度方向矩阵

:param step: Hough 变换步长大小

:param threshold: 筛选单元的阈值

\'\'\'

**def** Hough\_transform\_algorithm**(**self**):**

\'\'\'

:Usage: 按x,y,radius 建立三维空间，根据图中边上的点沿梯度方向对空间中所有单元进行投票。每个点投出来结果为折线。

:return: 投票矩阵

\'\'\'

**def** Select\_Circle**(**self**):**

\'\'\'

:Usage: 按阈值从投票矩阵中筛选合适的圆，并采用邻近点结果取平均值的方法作非极大化抑制

:return: None

\'\'\'

**def** Calculate**(**self**):**

\'\'\'

:Usage:按照算法顺序调用以上成员函数

:return: 圆形拟合结果图，圆的坐标及半径集合

\'\'\'

**（三）基于Hough算法的直线的检测**

基于Hough算法原理，可以复现出基于Hough算法的直线的检测。算法实现过程中的一些核心思想可概括如下，下面的报告中仅展示核心算法思想及部分代码，全部代码详见源文件。

（1）Hough变换实现直线检测算法的核心思想

对于圆弧检测的Hough算法可按照如下思路进行设计，详细算法详见代码设计

> Step1.量化与待检测图形有关的参数空间到合适精度
>
> Step2.初始化所有累加器为0
>
> Step3.对图像空间的每一点，在其所满足的参数方程对应的累加器上加1
>
> Step4.拥有最大值的累加器所在的坐标即为所求

（2）Hough变换实现圆检测算法的接口设计

在本次实验中，我设计了Hough\_Line\_Transform类以实现直线的检测，这个类的几个函数原型及各函数功能如下：

**class** **Hough\_Line\_Transform:**

**def** \_\_init\_\_**(**self**,** img**,** imgOrigin**):**

\'\'\'

:Usage: 初始化Hough\_Line\_Transform类

:param img: 输入的灰度图像

:param imgOrigin: 输入的RGB图像

\'\'\'

**def** voting**(**self**):**

\'\'\'

:Usage: 对图像空间的每一点，在其所满足的参数方程对应的累加器上加1

:Return: 投票矩阵

\'\'\'

**def** inverse\_hough**(**self**):**

\'\'\'

:Usage: 根据投票矩阵在图中绘制直线

:Return: 绘制直线后的图形

\'\'\'

**def** Calculate**(**self**):**

\'\'\'

:Usage:按照算法顺序调用以上成员函数

\'\'\'

五、实验结果与分析

本程序检测完成后共三张照片------利用Canny类进行边缘检测后的图像、利用Hough\_Circle\_Transform类进行圆检测后的图像、利用Hough\_Line\_Transform类进行直线检测后的图像，这些图像将直接存入与源程序同一目录下的Output文件夹内而不再在屏幕上，`Output`文件夹下展示了原始图片经过检测后的输出情况。

从上述输出结果可以看出，对于不同场景下的图片，本程序均能较好的检测到图片中的圆形和直线。具体来说，对于hw-highway图片，较好的检测出了图中左侧部分的直线，也很好的验证了图片中没有圆形这一关键信息；对于hw-seal图片，较好的检测出了图片中三个印章中的圆形，也较好地检测到了图中的直线；对于hw-coin图片，很好的检测出了图中的圆形图案，也很好的验证了图中没有直线这一关键信息。总的来说，本算法性能较好，能够较好的检测出图片中直线与圆形信息，但仍存在一定改进空间，这些改进空间具体体现在在部分超参数设计和算法设计上仍有一定优化空间，经过进一步优化后图像检测精度和速度可能会进一步提高。

Hough变换在检验已知形状的目标方面具有受曲线间断影响小和不受图形旋转的影响的优点，即使目标有稍许缺损或污染也能被正确识别。后期改进的基于概率的Hough变换提高了精准度与速度。常规Hough变换虽然具有显著的优势，但其不足也不容忽视------例如检测速度太慢，无法做到实时控制；精度不够高，期望的信息检测不到反而做出错误判断，进而产生大量的冗余数据。就圆检测而言，常规Hough变换的不足主要有以下几点：其一，参数由直线的两个参数，即截距和斜率，上升到三个，即圆心坐标和半径，每个点映射成参数空间的一个曲面，是一到多映射，因而计算量急剧增大；其二，需占用大量内存空间，耗时久、实时性差；其三，现实中的图像一般都受到外界噪声的干扰，信噪比较低，此时常规Hough变换的性能将急剧下降，进行参数空间极大值的搜索时由于合适的阈值难以确定，往往出现"虚峰"和"漏检"的问题。因而直接基于传统Hough算法基本原理所复现出的Hough检测函数运行速度较慢。

六、参考文献

\[1\] Chung C K L . An Efficient Randomized Algorithm for Detecting Circles\[J\]. Computer Vision and Image Understanding, 2001.

\[2\] Mohamed Roushdy. Detecting Coins with Different Radii based on Hough Transform in Noisy and Deformed Image, 2007.
