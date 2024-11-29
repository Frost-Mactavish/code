1.   DARE+TIES merge

     -   冻结backbone，融合Neck、RPN、RoI Head的参数
     -   扩展分类头的branch，是否选择原型进行微调
     
2.   zty GFSL, cls branch expansion

4.   蒸馏 → 软标签

     >软标签的一个侧面，类似于数据增强中的mixup方法
     >
     >在目标检测中，同样存在mixup和mosaic方法


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

​		Base training follows the implementation settings in BOSS, yet unable to reproduce the base 15 classes training results, which is 73.3 ~ 75.6 mAP@0.5.

| Backbone  | Phase | Old 10 | New 10 | mAP@0.5 |
| :-------: | :---: | :----: | :----: | :-----: |
| ResNet50  | joint | 69.97  | 72.17  |  71.07  |
|           | base  |   /    |   /    |  69.52  |
|           |  inc  |   /    |   /    |  72.89  |
| ResNet101 | joint | 71.54  | 74.31  |  72.93  |
|           | base  |   /    |   /    |  70.55  |
|           |  inc  |   /    |   /    |  73.49  |



## Incremental with Finetune

​		Tune different combo of components, with weight initialized from model trained on Old 10 classes.

<center><b>Tune with ResNet50 as backbone</b></center>

|    Component     | mAP@[0.5:0.95] | mAP@0.5 | Params | Comment |
| :--------------: | :------------: | :-----: | :----: | :-----: |
|       Head       |                |         | 15.26M |         |
|     Head+RPN     |                |         | 16.45M |         |
|   Head+RPN+FPN   |                |         | 19.79M |         |
| * w/ backbone4.2 |                |         | 24.26M |         |
|  * w/ backbone4  |                |         | 34.76M |         |
|       full       |                |         | 43.08M |         |

>`*` denotes the combination of `Head+RPN+FPN`
>
>`backbone4.2` denotes the final 3 conv layer of ResNet backbone
>
>`backbone4` denotes the 4th building block of ResNet backbone, abstrcted by pytorch

<center><b>Tune with ResNet101 as backbone</b></center>

|    Component     | mAP@[0.5:0.95] | mAP@0.5 | Params | Comment |
| :--------------: | :------------: | :-----: | :----: | :-----: |
|       Head       |                |         | 15.26M |         |
|     Head+RPN     |                |         | 16.45M |         |
|   Head+RPN+FPN   |                |         | 19.79M |         |
| * w/ backbone4.2 |                |         | 24.25M |         |
|  * w/ backbone4  |                |         | 34.76M |         |
|       full       |                |         | 62.07M |         |



## Incremental with Model Merge

|      | mAP@[0.5:0.95] | mAP@0.5 | Comment |
| :--: | :------------: | :-----: | :-----: |
|      |                |         |         |
|      |                |         |         |
|      |                |         |         |



## Incremental with LoRA

| Backbone  | Old 10 | New 10 | All  |     Params      | Comment |
| :-------: | :----: | :----: | :--: | :-------------: | :-----: |
| ResNet50  |        |        |      | 24.26M + (LoRA) |         |
| ResNet101 |        |        |      | 24.25M + (LoRA) |         |



## Issues

1.   Evaluation results over all classes are 0 after expanding classifer branches, presumably due to class misalignment

     >This occurs even when I keep the weight and architecture of the original model and just zero-pad the expanded branch of the classifier, which shouldnt have disrupted the class order.
     >
     >Another factor, that there could be misalignment of FPN mapping layer, could also contribute to an overall fall-behind.

2.   `loss_box_reg` resists to drop under current experiment settings, causing a major performance gap between current experiment and the one on pretrained weights

     >Training was nice and easy with pretained weights offered by PyTorch




## Paper Summary

#### Remote Sensing Object Detection Meets Deep Learning: A metareview of challenges and advances



#### Few-Shot Incremental Object Detection in Aerial Imagery via Dual-Frequency Prompt （TGRS24）



#### Balanced Orthogonal Subspace Separation Detector for Few-Shot Object Detection in Aerial Imagery （TGRS24）



#### Generalized few-shot object detection in remote sensing images (ISPRS23)



#### Incremental Detection of Remote Sensing Objects With Feature Pyramid and Knowledge Distillation （TGRS20）



#### Augmented Box Replay: Overcoming Foreground Shift for Incremental Object Detection (ICCV23)



#### Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation (CVPR22)

