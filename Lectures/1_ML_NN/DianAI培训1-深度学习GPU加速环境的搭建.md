# 深度学习GPU加速环境的搭建
## 贝贝组 黄涛
### 2018.07.23

---

# Install a software
### Learn to find tutorial from official website

---

# Install CUDA Toolkit
## First, search
![](/Users/hunto/Downloads/WX20180723-140740.png)

---

# Install CUDA Toolkit
[CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)
![](/Users/hunto/Downloads/WX20180723-141018.png)

---

# Install CUDA Toolkit
## NOTE: 
The latest CUDA toolkit version TensorFlow support officially is `9.0`
**DO NOT DOWNLOAD CUDA TOOLKIT `9.2`!**


![](/Users/hunto/Downloads/WX20180723-141139.png)

---

# Install CUDA Toolkit

![](/Users/hunto/Downloads/WX20180723-141506.png)

---


### Choose platform (exp. `Ubuntu 18.04`)
![70%](/Users/hunto/Downloads/WX20180723-141709.png)

---

### Follow official guide
1. Download deb (network) file.
2. Install deb package
  `sudo dpkg -i cuda-repo-ubuntu1704_9.0.176-1_amd64.deb`
3. Add keys
  `sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64/7fa2af80.pub`
4. Update software sources
  `sudo apt-get update`
6. Install cuda
  `sudo apt-get install cuda`

---

## Note
Ubuntu默认使用的Nvidia显卡驱动为开源驱动，安装CUDA会将驱动切换为Nvidia官方驱动

---
# Install cuDNN
The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. 

[Official site: https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

Download cuDNN, you need to join NVIDIA Developer Program.
![](/Users/hunto/Downloads/WX20180723-142538.png)

---

# Install cuDNN

![50%](/Users/hunto/Downloads/WX20180723-145811.png)
Just use `sudo dpkg -i` to install downloaded cuDNN deb file.

---

# Install TensorFlow-GPU

TensorFlow™ 是一个开放源代码软件库，用于进行高性能数值计算。借助其灵活的架构，用户可以轻松地将计算工作部署到多种平台（CPU、GPU、TPU）和设备（桌面设备、服务器集群、移动设备、边缘设备等）。TensorFlow™ 最初是由 Google Brain 团队（隶属于 Google 的 AI 部门）中的研究人员和工程师开发的，可为机器学习和深度学习提供强力支持，并且其灵活的数值计算核心广泛应用于许多其他科学领域。

Easiest way:
`pip install tensorflow-gpu`

Other ways:
[TensorFlow install: https://www.tensorflow.org/install/](https://www.tensorflow.org/install/)

---

# Install Pytorch-GPU
PyTorch is an optimized tensor library for deep learning using GPUs and CPUs.


[Official site: https://pytorch.org/](https://pytorch.org/)

Just follow the official guide:
![70%](/Users/hunto/Downloads/WX20180723-143328.png)

**Example: `Ubuntu18.04` `CUDA9.0` `Python3.6`**
```Shell
pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl`
pip3 install torchvision`
```


