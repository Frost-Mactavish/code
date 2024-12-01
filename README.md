1.   Knowledge Distillation → Soft Label

     >Soft Label is kinda like mix-up and mosaic in data augmentation
     >

[知乎：知识蒸馏insight](https://www.zhihu.com/question/309808462/answer/2649118476)

[数据增强Mixup原理与代码解读](https://blog.csdn.net/ooooocj/article/details/126070745)

[数据增强之Mosaic （Mixup,Cutout,CutMix）](https://zhuanlan.zhihu.com/p/405639109)

[数据增强-CutMix](https://zhuanlan.zhihu.com/p/516305068)

[一种巧妙且简单的数据增强方法 - MixUp 小综述](https://zhuanlan.zhihu.com/p/407089225)

[设置随机种子](https://www.zhihu.com/question/4070120479/answer/37568875429)

## Experiment Settings

| Method | Dataset |   Backbone   |  BS  |       optimizer       |               Scheduler                | Rounds |
| :----: | :-----: | :----------: | :--: | :-------------------: | :------------------------------------: | :----: |
| FPN-IL |  DIOR   |   ResNet50   |  2   |   SGD(2.5e-3, 0.9)    |   decays by 0.1 ervery 20k iteration   |  60k   |
|  BOSS  |  DIOR   |  ResNet101   |  16  | SGD(2e-2, 0.9, 1e-4 ) |     decays by 0.1 at iteration 15k     |  18k   |
|  ERD   |  COCO   |   ResNet50   |  16  | SGD(1e-2, 0.9, 1e-4)  | decays by 0.1 at iteration 60k and 80k |  90k   |
|  ABR   |   VOC   |   ResNet50   |  4   | SGD(5e-3, 0.9, 1e-4)  | decays by 0.1 at every 7.5k iteration  |  10k   |
|  Ours  |  DIOR   | ResNet50/101 |  8   | SGD(1e-2, 0.9, 1e-4)  |       decays by 0.1 at epoch 10        |  15e   |



## Base Training

Base training follows the implementation settings in BOSS, yet unable to reproduce the base 15 classes training results, which is 73.3 ~ 75.6 mAP@0.5.

This could be partially attributed to random seeding and sampler.

| Backbone  | Phase | Old 10 | New 10 | mAP@0.5 | mAP@[0.5:0.95] |
| :-------: | :---: | :----: | :----: | :-----: | :------------: |
| ResNet50  | joint |  70.0  |  72.2  |  71.1   |      44.3      |
|           | base  |   /    |   /    |  69.5   |      44.5      |
|           |  inc  |   /    |   /    |  72.9   |      42.7      |
| ResNet101 | joint |  71.5  |  74.3  |  72.9   |      45.5      |
|           | base  |   /    |   /    |  70.6   |      45.6      |
|           |  inc  |   /    |   /    |  73.5   |      42.8      |



## IOD with Fine-Tuning

Tune different combo of components, with weight initialized from model trained on Old 10 classes.

<center><b>Tune with ResNet50 as backbone</b></center>

|    Component     | mAP@0.5 | mAP@[0.5:0.95] | Params |
| :--------------: | :-----: | :------------: | :----: |
|       Head       |  31.9   |      16.5      | 15.26M |
|     Head+RPN     |  58.2   |      30.4      | 16.45M |
|   Head+RPN+FPN   |  69.2   |      39.6      | 19.79M |
| * w/ backbone4.2 |  71.2   |      41.5      | 24.26M |
|  * w/ backbone4  |         |                | 34.76M |
|       full       |  75.7   |                | 43.08M |

>`*` denotes the combination of `Head+RPN+FPN`
>
>`backbone4.2` denotes the final 3 conv layer of ResNet backbone
>
>`backbone4` denotes the 4th building block of ResNet backbone, abstrcted by pytorch

<center><b>Tune with ResNet101 as backbone</b></center>

|    Component     | mAP@0.5 | mAP@[0.5:0.95] | Params |
| :--------------: | :-----: | :------------: | :----: |
|       Head       |  34.6   |      16.8      | 15.26M |
|     Head+RPN     |  60.8   |      31.3      | 16.45M |
|   Head+RPN+FPN   |  71.8   |      40.8      | 19.79M |
| * w/ backbone4.2 |  72.1   |      41.5      | 24.25M |
|  * w/ backbone4  |         |                | 34.76M |
|       full       |  76.5   |      48.2      | 62.07M |

Now we can draw conclusion that **it achieves an optimal balance between performance and parameter-efficiency when RoI Head, RPN and FPN are tuned during new task learning**.



## IOD with Model Merge

|  Merge Method  | mAP@0.5 | mAP@[0.5:0.95] | Comment |
| :------------: | :-----: | :------------: | :-----: |
| Simple Average |         |                |         |
| Fisher Average |         |                |         |
|   Dare-Ties    |         |                |         |



## IOD with Low Rank Adaptation

`Low Rank Adaptation(LoRA)` is a Parameter Efficient Fine-Tuning(PEFT) technique based on matrix decomposition. LoRA approximates large weight matrix with low-rank matrices, achieving performance comparable to full-tuning with significantly fewer trainable parameters.

`Original LoRA`  decomposes arbitrary weight matrix into two low-rank matrices, name it linear, embedding or convolution layer, and later combines them with simple matrix element-wise addition.

A variant of LoRA, namely `Low-Rank Hadamard Product (LoHa)`, is similar to LoRA except it approximates the large weight matrix with more low-rank matrices and combines them with the Hadamard product. This method is even more parameter-efficient than LoRA and achieves comparable performance.

Besides, researchers have implemented LoRA modules particularly for convolution layers, named `ConvLoRA`.

<center><b>Original LoRA</b></center>

| Backbone  | Old 10 | New 10 | mAP@0.5 | mAP@[0.5:0.95] |     Params      |
| :-------: | :----: | :----: | :-----: | :------------: | :-------------: |
| ResNet50  |        |        |         |                | 24.26M + (LoRA) |
| ResNet101 |        |        |         |                | 24.25M + (LoRA) |

<center><b>Low-Rank Hadamard Product (LoHa)</b></center>

| Backbone  | Old 10 | New 10 | mAP@0.5 | mAP@[0.5:0.95] |     Params      |
| :-------: | :----: | :----: | :-----: | :------------: | :-------------: |
| ResNet50  |        |        |         |                | 24.26M + (LoRA) |
| ResNet101 |        |        |         |                | 24.25M + (LoRA) |

<center><b>ConvLoRA</b></center>

| Backbone  | Old 10 | New 10 | mAP@0.5 | mAP@[0.5:0.95] |     Params      |
| :-------: | :----: | :----: | :-----: | :------------: | :-------------: |
| ResNet50  |        |        |         |                | 24.26M + (LoRA) |
| ResNet101 |        |        |         |                | 24.25M + (LoRA) |



## Issues

1.   Evaluation results over all classes are 0 after expanding classifer branches, presumably due to class misalignment

     >This occurs even when I keep the weight and architecture of the original model and just zero-pad the expanded branch of the classifier, which shouldnt have disrupted the class order.
     >
     >Another factor, that there could be misalignment of FPN mapping layer, could also contribute to an overall fall-behind.

2.   `loss_box_reg` resists to drop under current experiment settings, causing a major performance gap between current experiment and the one on pretrained weights

     >Training was nice and easy with pretained weights offered by PyTorch

3.   Gotta revisit papers read before.




## Paper Summary

#### Remote Sensing Object Detection Meets Deep Learning: A metareview of challenges and advances



#### Few-Shot Incremental Object Detection in Aerial Imagery via Dual-Frequency Prompt (TGRS24)



#### Balanced Orthogonal Subspace Separation Detector for Few-Shot Object Detection in Aerial Imagery (TGRS24)



#### Generalized few-shot object detection in remote sensing images (ISPRS23)



#### Incremental Detection of Remote Sensing Objects With Feature Pyramid and Knowledge Distillation (TGRS20)



#### Augmented Box Replay: Overcoming Foreground Shift for Incremental Object Detection (ICCV23)



#### Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation (CVPR22)

