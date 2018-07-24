
# Dian AI培训
### 贝贝组 黄涛
#### 2018.07.23

---

# Contents
* Machine Learning （机器学习）
* Deep Learning （深度学习）
* Deep Reinforcement Learning （深度强化学习）
* Homework

---

# How to learn machine learning?
## Courses:
* [Machine Learning - Adrew Ng](https://www.coursera.org/learn/machine-learning)
* [CS231n - Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
* [CS224n - Natural Language Processing with Deep Learning](http://cs224n.stanford.edu/)
  * [CS224n课程笔记-Hunto](https://hunto.github.io/tags.html#CS224n)

## Books:
* 机器学习-周志华
* Deep Learning - Ian Goodfellow and Yoshua Bengio and Aaron Courville
  * en: [http://www.deeplearningbook.org/](http://www.deeplearningbook.org/)
  * cn: [https://github.com/exacity/deeplearningbook-chinese](https://github.com/exacity/deeplearningbook-chinese)

---

## Relationship between AI,ML,DL


* AI: Artificial Intelligence
* ML: Machine Learning
* DL: Deep Learning

![80%](/Users/hunto/Downloads/114785133_17_20171029035307304.jpg)

---

# Machine Learning ≈ Looking for a Function
* Speech Recognition
![](/Users/hunto/Downloads/WX20180722-155151.png)

* Image Recognition
![70%](/Users/hunto/Downloads/WX20180722-155234.png)

* Playing Go
![70%](/Users/hunto/Downloads/WX20180722-155318.png)

* Dialogue System
![70%](/Users/hunto/Downloads/WX20180722-155337.png)

---

# Types of Machine Learning
![40%](/Users/hunto/Downloads/WX20180721-173109.png)

* Supervised Model(监督学习模型) : 任务驱动
  训练数据带有标签
  * Regression
  * Classification
* Unsupervised Model(无监督学习模型)：数据驱动
  训练数据无标签
  * Clustering
* Reinforcement(强化学习)：学习适应环境 

---

# Framework
A set of function: $f_1, f_2, ..., f_n$

**example**
Image Recognition
![70%](/Users/hunto/Downloads/WX20180722-155747.png)

Functions:
![70%](/Users/hunto/Downloads/WX20180722-155829.png)

---

# Framework
![](/Users/hunto/Downloads/WX20180722-160022.png)

## Supervised Learning
* We have **training data**.
* We have **a set of function**.
* $f(input)=f_{out}$，$cmp(real\ data,f_{out})=>f_{score}$
* Pick the "best" function.

---

## Supervised learning

![](/Users/hunto/Downloads/WX20180722-161228.png)

---

## How to train a supervised model?
1. Find an appropriate cost function for your task
2. Minimize the cost function

**Example**
* Square loss for linear regression
$$C=\sum^n_{i=1}(y_i-\hat y_i)^2$$

---

## Minimize cost function
* Gradient descent
$$\theta_j=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)$$

![](/Users/hunto/Downloads/WX20180722-202137.png)

---

## How to evaluation?
### 3 situations
![](/Users/hunto/Downloads/WX20180722-202344.png)

---

## How to evaluation?
1. Split train and test dataset
    train: usually $\frac23$~$\frac45$ of data
2. Train model on train dataset, and using test dataset to evaluation the model.
3. Evaluation methods:
    * Accuracy（准确率）
    * Recall（召回率）
    * F-score（加权调和平均）: $F=\frac{(a^2+1)P\cdot R}{a^2(P+R)}$
      $a=1 => F_1 = \frac{2\cdot P\cdot R}{P+R}$
    * AUC
    * ... 
---

# ML Pipeline
* Data collecting
* Feature engineering
* Model building
* Application
* Feedback and Optimization

---

# Feature Engineering
* Feature is the limitation of model.
  **70%** time for feature engineering, **30%** time for modeling
* Manual features
  * Statistical features
  * Time series
  * Prior knowledge

### Example: stock risk prediction

---

# Model

### Classification Model
* Logistic regression
* SVM (Supported Vector Machine)
* Decision Tree

### Regression Model
* Linear regression
* Non-linear regression

### Clustering Model
* K-means

---

## Linear regression （线性回归）
![60%](/Users/hunto/Downloads/213.png)
$$g(x)=w_0 + w_1x_1+...+w_nx_n$$

---

## Logistic Regression （逻辑回归）

![70%](/Users/hunto/Downloads/WX20180721-190405.png)

$g(x)=w_0 + w_1x_1+...+w_nx_n$
$sigmoid(x) = \frac{1}{1+e^{-x}}$

$P(y=1|x)=sigmoid (g(x)) = \frac{1}{1+e^{-g(x)}}$



---

## Decision Tree （决策树）
决策树是运用于分类的一种树结构，其中的每个内部节点代表对某一属性的一次测试，每条边代表一个测试结果，叶节点代表某个类或类的分布。
决策树的决策过程需要从决策树的根节点开始，待测数据与决策树中的特征节点进行比较，并按照比较结果选择选择下一比较分支，直到叶子节点作为最终的决策结果。
![130%](/Users/hunto/Downloads/162042244705421.jpg)

---

# Deep Learning


### What's deep learning
Deep learning is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called artifical neural networks.

### Why deep learning is so hot?
* Coming up with features is difficult, time-consuming, requires expert knowledge. "Applied machine learning" is basically feature engineering. (Andrew Ng)
* But deep learning can automatically extract features!

---

## 传统机器学习的局限
![](/Users/hunto/Downloads/1530935881916-ff2d6a30-d87c-4d73-9cc9-b1c214548dbb-image-resized.png)
Softmax(logistic regression)只有线性的决策边界，分类效果有限，当问题变得复杂时，效果不好， 但神经网络可以学到复杂得多的特性和非线性决策边界。

---

## Deep learining applications
* Computer Vison:
  * Image Recognition
* Speech Recognition
* Natural Language Processing
  * Machine translation
  * Sentiment analysis
  * Text classification 

---

## From logistic regression to neural networks
神经网络的每一个神经元都是一个二分类逻辑回归单元。
![70%](/Users/hunto/Downloads/1530944741741-e223c4df-4a17-4979-8abd-80e72b4b58f8-image.png)

---

![](/Users/hunto/Downloads/1530944805095-1b7f3492-aef5-4a5b-a062-1c4c33f6e8cd-image.png)

---

## Multi-layer Neural Networks
![80%](/Users/hunto/Downloads/WX20180722-213058.png)

---

## Activation function
![50%](/Users/hunto/Downloads/31133132.png)

**Why activation function must be `non-linear`?**
多层线性系统叠加仍然是线性系统。
$x\cdot W_1\cdot W_2\cdot ... \cdot W_n = x \cdot W$

---

## Feed-forward Computation
![50%](/Users/hunto/Downloads/1530945475411-c42b3315-ed99-4068-8f2d-62a03919a864-image-resized.png)

$z=Wx+b$w
$a = f(z)$
$s=U^Ta$

---

## Feed-forward computation
**Example**
1. We have $x=[0, 1,2,3]$
2. $z=Wx+b$, suppose $W_1=2,b_1=1,W_2=1,b=-1$
$z_1=[1,3,5,7],z_2=[-1,0,1,2]$
3. Activation function, exp. : Softmax $f(x)=\frac{1}{1+e^{-x}}$
$a_1=[0.731,0.952,0.993,0.999]$
$a_2=[0.268, 0.5,0.731,0.880]$
4. $s=U^Ta$, suppose $U=[0.4, 0.6]$
$s = [0.4,0.6]^T\cdot[a_1,a_2] = [0.453,0.680,0.836,0.928]$

---

## Loss Function

### softmax & cross-entropy loss(交叉熵损失)
* softmax
$$softmax(x)_i=\frac{exp(x_i)}{\sum_jexp(x_j)}$$
* cross-entropy loss
$$CE(y,\hat y)=-\sum^{|V|}_{j=1}y_jlog\hat y_j$$
* softmax cross-entropy loss
$$J(\theta) = -\sum^{|V|}_{j=1}y_jlog(softmax(s_j))$$

---

## Loss Function
**Example**
1. We have $x=[0, 1,2,3]$
2. $z=Wx+b$
3. Activation function, exp. : Softmax $f(x)=\frac{1}{1+e^{-x}}$
4. $s=U^Ta$
$s =[0.453,0.680,0.836,0.928]$
5. $\hat y = softmax(s)$
$\hat y_1 = [0.323, 0.257,0.220,0.201]$
6. Loss $J=-\sum^{|V|}_{j=1}y_jlog(softmax(s_j))$, suppose $y=[1,0,0,0]$
$e = 1.130$

---

## Backpropagation（反向传播）
### Multi-layer neural networks
![](/Users/hunto/Downloads/1530956303750-65c79f5d-d8bb-4f4c-b924-6eff9a077c26-image-resized.png)

---

## Backpropagation
**Why we need multi-layer nets?**
层数越多，可以表达的问题越复杂
![](/Users/hunto/Downloads/1530956404164-7a2057de-cb9b-40dc-a847-807ee2ec4d63-image.png)

---

## Feed-forward & Backpropagation
![](/Users/hunto/Downloads/1530956820599-f6caaaa7-e4d3-453c-88d8-927dcc21f193-image.png)

**Why?**
* **Chain Rule**（**链式法则**）
$$(g\circ f)'(x)=[g(f(x))]'=g'(f(x))f'(x)=\frac{du}{dx}\cdot \frac{dy}{dx}$$

---

## Backpropagation


![50%](/Users/hunto/Downloads/1530956884399-16b2e385-014a-44e5-ab6d-fdac43e34d23-image.png)
**Chain Rule**
$u=g(y_1,y_2,...,y_n),y=f(x_1,x_2,...,x_n)$
$$\frac{\partial u}{\partial x_i} = \sum^n_{j=1}\frac{\partial g}{\partial y_j}\cdot \frac{\partial y_j}{\partial x_i}$$

---

## Optimizer
* SGD (Stochastic Gradient Descent / Mini-batch Gradient Descent)
  * Gradient Descent in a mini batch(a part of trainset) 
* Momentum	90
* Adam(Adaptive Moment Estimation)

---

## Review

Steps to train a neural network:
* Process train data: data to vector, split train/dev data.
* Initialize network weights.
* Feed-forward computation: input x -> output score.
* Cost computation: score vs. real y -> error.
* Backpropagation: propagate loss back. 
* Optimizer: update weights.

---

# Deep Reinforcement Learning

Closest to Artificial General Intelligence(AGI)

* It's the learning paradigm of biological.
* RL provides the resource of data.
* DL automatically extract features.
* When there's a general method for machine to build models, AGI will come.

---

# Homework
* 深入了解本次培训的内容，如神经网络、激活函数等
* 熟悉深度学习框架Pytorch
  * [Pytorch Tutorials](http://pytorch.org/tutorials/)
  * [Pytorch 中文教程](http://pytorch.apachecn.org/cn/tutorials/index.html)
* 使用Pytorch实现一个全连接神经网络，对[MINIST数据集](http://yann.lecun.com/exdb/mnist/)进行分类，有余力的同学可以尝试在[cifar-10](http://www.cs.toronto.edu/~kriz/cifar.html)上的效果。