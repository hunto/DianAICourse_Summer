# DianAI培训2
## Convolutional Neural Networks
### 贝贝组 黄涛
#### 2018.7.26

---

# Neural Networks Review
Steps to train a neural network:
* Process train data: data to vector, split train/dev data.
* Initialize network weights.
* Feed-forward computation: input x -> output score.
* Cost computation: score vs. real y -> error.
* Backpropagation: propagate loss back. 
* Optimizer: update weights.

---

# Convolutional Neural Networks
## What's CNN
* 2D-Convolution
* 2D-Convolution with Multi-Channels
* Pooling

---

## CNN - Better than single layer fc NN in MNIST
![](/Users/hunto/Downloads/WX20180724-211759@2x.png)


---

## 2D-Convolution
---

![](/Users/hunto/Downloads/WX20180724-165154.png)

---

![](/Users/hunto/Downloads/WX20180724-165259.png)

---

![](/Users/hunto/Downloads/WX20180724-165320.png)

---

![](/Users/hunto/Downloads/WX20180724-165413.png)

---

![](/Users/hunto/Downloads/WX20180724-165337.png)

---

![](/Users/hunto/Downloads/WX20180724-165430.png)

---

![](/Users/hunto/Downloads/WX20180724-165626.png)

---

![](/Users/hunto/Downloads/WX20180724-165647.png)

---

![](/Users/hunto/Downloads/WX20180724-165659.png)

---

## 2D-Convolution
Different convolution kernels can extract different features.
![70%](/Users/hunto/Downloads/123141.png)
![70%](/Users/hunto/Downloads/67543.png)

---

## 2D-Convolution with Multi-Channels
![](/Users/hunto/Downloads/6453.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RGB channel &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;adding&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;activation&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;feature map

$$$$

* Use convolution kernel on N channels and generate N feature maps
* Add N feature maps to one 
* Non-linear activation function

---

## Covolutional Neural Networks

![](/Users/hunto/Downloads/dl_3_1.png)


---

## Pooling 

![60%](/Users/hunto/Downloads/a2.png)

Why pooling?
* invariance, increase generalization ability
* reduce parameters
* gain output of fixed length


---

### Pooling methods
* max pooling (better on texture)
  * forward: $[[1, 3], [2, 2]] -> [3]$
  * backward: $[3] -> [[0, 3], [0, 0]]$
* average pooling (better on background)
  * forward: $[[1, 3], [2, 2]] -> [2]$
  * backward: $[2] -> [[0.5, 0.5], [0.5, 0.5]]$
* ...

---

## Convolutional Neural Networks
![](/Users/hunto/Downloads/WX20180724-171709.png)
![](/Users/hunto/Downloads/WX20180724-171739@2x.png)

---

## Convolutional Neural Networks
CNN works better than full-connected neural networks.

**Why?**
1. Biology and neuroscience.
   CNN can perform like our eyes.
2. Convolution works well on extracting features from images.
3. Parameters sharing and Local Connection.
  **Example:**
  Image $28\times 28$
  Using two dense layers of 512 nodes and 128 nodes,
  parameter size is $28 \times 28 \times 512+ 512\times 128$
  Using two conv layers of 512 kernels and 256 kernels,
  parameter size is $512\times 3 \times 3+256\times 3 \times 3$

---

## Convolutional Neural Networks
![](/Users/hunto/Downloads/453.png)

---

# Image Net
ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. 

* 1998: LeNet (origin) [Gradient-Based Learning Applied to Document Recognition]
* 2012: AlexNet (revolution) [ImageNet Classiﬁcation with DeepConvolutional Neural Networks]
* 2014: VGG-net (stable) [Very Deep Concolutional Networks For Large-Scale Image Recognition]
* 2014: GoogleNet (Deep) [Going deeper with convolutions]
* 2015: ResNet (extraordinarily deep) [Deep Residual Learning for Image Recognition]
* 2017: SEnet (End) [Squeeze-and-Excitation Networks]

---

# ImageNet
![](/Users/hunto/Downloads/5432.png)
<center>Image Classification</center>
<center>Object Localization</center>
<center>Object Detection</center>
<center>more than 15,000,000 images</center>

---

## CNN in ImageNet
![](/Users/hunto/Downloads/6543.png)

---

## CNN Model - LeNet

![](/Users/hunto/Downloads/534224.png)

* `Input`: 32x32
* Conv layer `C1`: 
  * `kernel size` = $5\times5$
  * `conv num` = $6$
  * `output` = $28\times28$ (32-5+1)
  * `W size` = $28\times28\times6$
  * `trained params` = $(5\times 5+1)\times6=156$ (25 w, 1 bias)
  * `connections` = $(5\times5+1)\times6\times28\times28=122304$
---

## CNN Model - LeNet

![](/Users/hunto/Downloads/534224.png)

* Subsampling layer`S2`: 
  * `sampling input size` = $2\times 2$
  * `sampling type`: sum(input) * w + b
  * `output size`: $14\times14$
  * `trained params`: $2\times6=12$
  * `connections`: $(2\times2+1)\times6\times14\times14=5880$

---

## CNN Model - LeNet

![](/Users/hunto/Downloads/534224.png)

* Conv layer `C3`:
  * `kernel size` = $5\times5$
  * `conv num` = $16$
  * `output` = $10\times10$
  ![100%](/Users/hunto/Downloads/dl_3_5.png)

---
 
## CNN Model - LeNet

![](/Users/hunto/Downloads/534224.png)

 * Subsampling layer `S4`：
   * `output` = $16\times5\times5$ 
 * Conv layer `C5`(same as fc):
   * `kernel size` = $5\times5$
   * `conv num` = $120$
   * `output` = $120\times1\times1$
  
---

## CNN Model - LeNet

![](/Users/hunto/Downloads/534224.png)

* Full connection layer `F6`:
  * `input` = $120\times1$
  * `output` = $84$

* Full connection layer`OUTPUT`:
  * `input` = $84 \times 1$
  * `output` = $10\times1$

---

### CNN Model - AlexNet
![](/Users/hunto/Downloads/34445354.png)

---

### Data Augmentation
![](/Users/hunto/Downloads/WX20180724-204141.png)

* Rotation
* Reflection
* Flip
* Zoom
* Shift
* Scale
* Constrast
* Noise

---

### VGG-net
![50%](/Users/hunto/Downloads/876543.png)

* smaller kernel (3x1, 1x1)
* more conv layers with a pooling layer
* deeper

---

# Homework
* 深入复习本次培训内容，理解卷积神经网络
* 使用Pytorch实现卷积神经网络，比较其在cifar-10上的效果
