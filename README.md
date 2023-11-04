# StereoNet_nfyg

如果喜欢/有帮助,请关注 BiliBili 逆风引弓 https://space.bilibili.com/212529

If you like it/find it helpful, please follow BiliBili 逆风引弓 https://space.bilibili.com/212529

============================================================================================================

一个双目深度估计模型

A stereo depth estimation model

基于文献 "Stereonet-Guided hierarchical refinement for real-time edge-aware depth prediction"，同时也参考了其它github项目的代码

based on paper "Stereonet-Guided hierarchical refinement for real-time edge-aware depth prediction", also refers to other github projects code


训练采用sceneflow数据集练了40epoch

training set is sceneflow dataset with 40 epoch

平均EPE误差约1.5像素

average EPE(end point error) is ~1.5px

在我个人电脑的3060Ti上跑540*960分辨率的图片，平均一张35ms，试着做量化以便于边缘部署，但是这类对精度要求高的回归模型，量化很难，况且其中有各种非常规算子，例如创建cost volume，线性插值上采样，量化效果不太好，精度损失严重，当然也可能是我不太懂量化的技术吧
