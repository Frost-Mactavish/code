## 代码使用方法

系统环境：Ubuntu 22.04 LTS，CUDA 12.4，PyTorch 2.2

更多实验环境信息在报告里说明

1.   根据ViT.py中的注释对PyTorch源码进行修改

     >对PyTorch实现中VisionTransformer类的类方法forward进行改动，通过PyTorch内置Profiler类监视 x = self.encoder(x) 运行时间
     >
     >**如果仅需要验证CUDA C的有效性，可跳过这一步**

2.   运行compile.sh脚本

3.   生成编译文件ViT-CUDA

4.   相关输出被重定向到result.txt中

     >result.txt输出内容说明：
     >
     >最后三行为ViT.cu输出的BatchSize、代码总运行时间和GPU内核时间
     >
     >其他部分是PyTorch Profiler的输出，其中Self CUDA time total为PyTorch Profiler记录到的GPU内核时间