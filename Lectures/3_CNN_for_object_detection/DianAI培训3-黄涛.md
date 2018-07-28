# DianAI培训3
## CNN for Object Localization and Detection
### 贝贝组 黄涛
#### 2018.07.30

---

# Object Localization and Detection
![](/Users/hunto/Downloads/WX20180725-151616.png)

---

## Computer Vision Tasks
![](/Users/hunto/Downloads/WX20180725-151716.png)

---

## Computer Vision Tasks
![](/Users/hunto/Downloads/WX20180725-151750.png)

---

## Classification + Localization: Task
![](/Users/hunto/Downloads/WX20180725-151950.png)

---

## Classification + Localization: ImageNet
![](/Users/hunto/Downloads/WX20180725-152138.png)

---

## Idea #1: Localization as Regression
![](/Users/hunto/Downloads/WX20180725-152233.png)

---

## Simple Recipe for Classification + Localization
**Step 1**: Train (or download) a classification model (AlexNet, VGG, Google Net)

![](/Users/hunto/Downloads/WX20180725-152507.png)

---

## Simple Recipe for Classification + Localization
**Step 2**: Attach new fully-connected "regression head" to the network

![](/Users/hunto/Downloads/WX20180725-152616.png)

---

## Simple Recipe for Classification + Localization
**Step 3**: Train the regression head only with SGD and L2 loss

![](/Users/hunto/Downloads/WX20180725-152733.png)

---

## Simple Recipe for Classification + Localization
**Step 4**: At test time use both heads

![](/Users/hunto/Downloads/WX20180725-152845.png)

---

## Per-class vs class agnostic regression

![](/Users/hunto/Downloads/WX20180725-152952.png)

---

## Where to attach the regression head?
![](/Users/hunto/Downloads/WX20180725-153136.png)

---

## Aside: Localizing multiple objects
![](/Users/hunto/Downloads/WX20180725-153300.png)

---

## Aside: Human Pose Estimation
![](/Users/hunto/Downloads/WX20180725-153346.png)

---

## Localization as Regression
* Very simple
* Think if you can use this for projects

---

## Idea #2: Sliding Window
* Run classification + regression network at multiple locations on a high-resolution image
* Convert fully-connected layer into convolutional layers for efficient computation
* Combine classifier and regressor predictions across all scales for final prediction

---

## Sliding Window: Overfeat
![](/Users/hunto/Downloads/WX20180725-153843.png)

---

## Sliding Window: Overfeat
![](/Users/hunto/Downloads/WX20180725-154057.png)

---

## Sliding Window: Overfeat
![](/Users/hunto/Downloads/WX20180725-154236.png)

---

## Sliding Window: Overfeat
![](/Users/hunto/Downloads/WX20180725-154338.png)

---

## Sliding Window: Overfeat
![](/Users/hunto/Downloads/WX20180725-154417.png)

---

## Sliding Window: Overfeat
![](/Users/hunto/Downloads/WX20180725-154533.png)

---

## Sliding Window: Overfeat
![](/Users/hunto/Downloads/WX20180725-154601.png)

---

## Sliding Window: Overfeat
![](/Users/hunto/Downloads/WX20180725-154627.png)

---

## Sliding Window: Overfeat
In practice use many sliding window locations and multiple scales.
![](/Users/hunto/Downloads/WX20180725-154704.png)


---

## Efficient Sliding Window: Overfeat
![](/Users/hunto/Downloads/WX20180725-154822.png)

---

## Efficient Sliding Window: Overfeat
![](/Users/hunto/Downloads/WX20180725-154856.png)

---

## Efficient Sliding Window: Overfeat
![](/Users/hunto/Downloads/WX20180725-154939.png)


---

## ImageNet Classification + Localization
![](/Users/hunto/Downloads/WX20180725-155116.png)

---

## Computer Vision Tasks
![](/Users/hunto/Downloads/WX20180725-151750.png)


---

## Computer Vision Tasks
![](/Users/hunto/Downloads/WX20180725-155213.png)


---

## Detection as Regression?

![](/Users/hunto/Downloads/WX20180725-155257.png)

---

## Detection as Regression?
![](/Users/hunto/Downloads/WX20180725-155320.png)

---

## Detection as Regression?
![](/Users/hunto/Downloads/WX20180725-155356.png)

---

## Detection as Classification
![](/Users/hunto/Downloads/WX20180725-155517.png)

---

## Detection as Classification
![](/Users/hunto/Downloads/WX20180725-155546.png)

---

## Detection as Classification
![](/Users/hunto/Downloads/WX20180725-155620.png)

---

## Detection as Classification
**Problem**: Need to test many positions and scales.
**Solution**: If your classifier is fast enough, just do it.

---

## Histogram of Oriented Gradients
### 方向梯度直方图(HOG)
![](/Users/hunto/Downloads/WX20180725-155814.png)

---

## Deformable Parts Model (DPM)

![](/Users/hunto/Downloads/WX20180725-160220.png)

---

## Aside: Deformable Parts Models are CNNs?
![](/Users/hunto/Downloads/WX20180725-160312.png)

---

## Detection as Classification
**Problem**: Need to test many positions and scales, and use a computationally demanding classifier (CNN).
**Solution**: Only look at a tiny  subset of possible positions.

---

## Region Proposals
* Find "blobby" image regions that are likely to contain objects.
* "Class-agnostic" object detector.
* Look for "blob-like" regions

![](/Users/hunto/Downloads/WX20180725-160942.png)

---

## Region Proposals: Selective Search
![](/Users/hunto/Downloads/WX20180725-161042.png)

---

## Region Proposals: Many other choices
![](/Users/hunto/Downloads/WX20180725-161109.png)

---

## Region Proposals: Many other choices
![](/Users/hunto/Downloads/WX20180725-161211.png)

---

## Putting it together: R-CNN
![](/Users/hunto/Downloads/WX20180725-161311.png)

---

## R-CNN Training
**Step 1**: Train (or download) a classification model for ImageNet(AlexNet).
$$$$
![](/Users/hunto/Downloads/WX20180725-161502.png)

---

## R-CNN Training
**Step 2**: Fine-tune model for detection
* Instead of 1000 ImageNet classes, want 20 object classes + background
* Throw away final fully-connected layer, reinitialize from scratch.
* Keep training model using positive / negative regions from detection images.

![](/Users/hunto/Downloads/WX20180725-161723.png)


---

## R-CNN Training
**Step 3**: Extract features
* Extract region proposals for all images.
* For each region: warp to CNN input size, run forward through CNN, save pool5 features to disk.
* Have a big hard drive: features are ~200GB for PASCAL dataset!

![](/Users/hunto/Downloads/WX20180725-161924.png)

---

## R-CNN Training
**Step 4**: Train one binary SVM per class to classify region features.

![](/Users/hunto/Downloads/WX20180725-162024.png)

---

## R-CNN Training
**Step 4**: Train one binary SVM per class to classify region features.

![](/Users/hunto/Downloads/WX20180725-162123.png)


---

## R-CNN Training
**Step 5**: (bbox regression): For each class, train a linear regression model to map from cached features to offsets to GT boxes to make up for "slightly wrong" proposals.

![](/Users/hunto/Downloads/WX20180725-162336.png)

---

## Object Detection: Datasets
![](/Users/hunto/Downloads/WX20180725-162424.png)

---

## Object Detection: Evaluation
* We use a metric called "mean average precision" (mAP) .
* Compute average precision (AP) separately for each class, then average over classes.
* A detection is a true positive if it has loU with a ground-truth box greater than some threshold (usually 0.5) (mAP@0.5).
* Combine all detections from all test images to draw a precision / recall curve for each class; AP is a area under curve.
* TL;DR mAP is a number from 0 to 10; higher is better.

---

## R-CNN Results
![](/Users/hunto/Downloads/WX20180725-163002.png)

---

## R-CNN Results 
Big improvement compared to pre-CNN methods.

![](/Users/hunto/Downloads/WX20180725-163046.png)

---

## R-CNN Results
Bounding box regression helps a bit.

![](/Users/hunto/Downloads/WX20180725-163210.png)

---

## R-CNN Results
Features from a deeper network help a lot.

![](/Users/hunto/Downloads/WX20180725-163300.png)

---

## R-CNN Problems
* Slow at test-time: need to run full forward pass of CNN for each region proposal.
* SVMs and regressions are post-hoc: CNN features not updated in response to SVMs and regressors.
* Complex multistage training pipeline.

---

![](/Users/hunto/Downloads/WX20180725-163558.png)

---

![](/Users/hunto/Downloads/WX20180725-163615.png)

---

## Fast R-CNN: Region of Interst Pooling
![](/Users/hunto/Downloads/WX20180725-163709.png)

---

## Fast R-CNN: Region of Interst Pooling
![](/Users/hunto/Downloads/WX20180725-163740.png)

---

## Fast R-CNN: Region of Interst Pooling

![](/Users/hunto/Downloads/WX20180725-163759.png)

---

## Fast R-CNN: Region of Interst Pooling

![](/Users/hunto/Downloads/WX20180725-163827.png)

---

## Fast R-CNN: Region of Interst Pooling
![](/Users/hunto/Downloads/WX20180725-163842.png)

---

## Fast R-CNN Results

![](/Users/hunto/Downloads/WX20180725-164013.png)

<center>Using VGG-16 CNN on Pascal VOC 2007 dataset</center>

---

## Fast R-CNN Results

![](/Users/hunto/Downloads/WX20180725-164152.png)

<center>Using VGG-16 CNN on Pascal VOC 2007 dataset</center>

---

## Fast R-CNN Results

![](/Users/hunto/Downloads/WX20180725-164234.png)

<center>Using VGG-16 CNN on Pascal VOC 2007 dataset</center>

---

## Fast R-CNN Problem
Test-time speeds don't include region proposals.

![](/Users/hunto/Downloads/WX20180725-164343.png)

---

## Fast R-CNN ~~Problem~~ Solution
Test-time speeds don't include region proposals.
Just make the CNN do region proposals too!

![](/Users/hunto/Downloads/WX20180725-164422.png)

---

![](/Users/hunto/Downloads/WX20180725-164551.png)

---

## Faster R-CNN: Region Proposal Network
![](/Users/hunto/Downloads/WX20180725-164615.png)

---

## Faster R-CNN: Region Proposal Network
![](/Users/hunto/Downloads/WX20180725-164658.png)

---

## Faster R-CNN: Training
 
![](/Users/hunto/Downloads/WX20180725-164935.png)

---

## Faster R-CNN: Results

![](/Users/hunto/Downloads/WX20180725-164959.png)

---

## Object Detection State-of-the-art:
## ResNet 101 + Faster R-CNN + some extras

![](/Users/hunto/Downloads/WX20180725-165113.png)

---

## ImageNet Detection 2013-2015
![](/Users/hunto/Downloads/WX20180725-165131.png)

---

![](/Users/hunto/Downloads/WX20180725-165245.png)

---

![](/Users/hunto/Downloads/WX20180725-165302.png)

---

![](/Users/hunto/Downloads/141421.png)

---

![](/Users/hunto/Downloads/654.png)

---

![](/Users/hunto/Downloads/WX20180725-165355.png)

---

![](/Users/hunto/Downloads/WX20180725-165412.png)

---

![](/Users/hunto/Downloads/WX20180725-165434.png)

---

![](/Users/hunto/Downloads/WX20180725-165449.png)

---

![](/Users/hunto/Downloads/WX20180725-165504.png)

---

## Object Detection code links:
* R-CNN
(Caffe + MATLAB): [https://github.com/rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)
Probably don't use this; Too slow.

* Fast R-CNN
(Caffe + MATLAB): [https://github.com/rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)

* Faster R-CNN
(Caffe + MATLAB): [https://github.com/ShaoqingRen/faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn)
(Caffe + Python): [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

* YOLO
[http://pjreddie.com/darknet/yolo](http://pjreddie.com/darknet/yolo)

---

# Recap
## Localization
* Find a fixed number of objects (one or many)
* L2 regression from CNN features to box coordinates
* Much simpler than detection; consider it for your projects!
* Overfeat: Regression + efficient sliding window with FC -> conv conversion
* Deeper networks do better

---

# Recap
## Object Detection
* Find a variable number of objects by classifying image regions
* Before CNNs: dense multiscale sliding window (HoG, DPM)
* Avoid dense sliding window with region proposals
* R-CNN: Selective Search + CNN classification / regression
* Fast R-CNN: Swap order of convolutions and region extraction
* Faster R-CNN: Compute region proposals within the network
* Deeper networks do better