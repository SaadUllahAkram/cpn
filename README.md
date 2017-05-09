# Cell Proposal Network (CPN)

1. **[Introduction](#introduction)**
1. **[Requirements](#requirements)**
1. **[Instructions](#instructions)**

## [Introduction:](#introduction)
This repository contains code for proposing candidate masks for biological cells in microscopy images. These cell candidates are used for cell tracking. The paper can be found at **[arXiv](https://arxiv.org/abs/.)** and tracking code is available at: **[Cell Tracker](https://github.com/SaadUllahAkram/CellTracker)**

If you find this code useful in your research, please cite:

    @article{akram2017a,
        author = {Akram S. U., Kannala J., Eklund L., and Heikkilä J.},
        title = {Cell Tracking via Proposal Generation and Selection},
        journal = {arXiv:xxxx.xxxx},
        year = {2017}
    }

This code is based on [Faster R-CNN](https://github.com/ShaoqingRen/faster_rcnn) and many functions from original code were either used directly or with slight modifications.
The code has many half finished and broken features, which i plan to either remove or fix in future.
The code was re-structred recently and it may have introduced some bugs, which you may report (especially if they are in parts which are executed) and I will try to fix them.
If and when I fix this code, I may remove the experimental **exp** branch.

## [Requirements:](#requirements)
1. [CPN](https://github.com/SaadUllahAkram/CPN): Cell Proposal Network code.<br/>
    `git clone https://github.com/SaadUllahAkram/CPN.git`
1. [caffe](https://github.com/SaadUllahAkram/caffe_cpn): Faster R-CNN version with crop layer.<br/>
    `git clone https://github.com/SaadUllahAkram/caffe_cpn.git`
1. [BIA](https://github.com/SaadUllahAkram/BIA): a collection of useful functions.<br/>
    `git clone https://github.com/SaadUllahAkram/BIA.git`
1. [MATLAB](https://www.mathworks.com/products/matlab.html)<br/>

## [Instructions:](#instructions)
1. Set `Caffe, ISBI CTC data` paths in `get_paths.m` function in [BIA](https://github.com/SaadUllahAkram/BIA):
1. activate caffe using: `bia.caffe.activate('cpn', gpu_id);`

#### Testing (will be added soon)
1. run `demo_test_cpn()`

#### Training
1. Download [Cell Tracking Challenge](http://www.codesolorzano.com/Challenges/CTC/Welcome.html) data.
1. run `demo_train_cpn()`

