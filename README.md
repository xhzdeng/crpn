# Corner-based Region Proposal Network

CRPN is a two-stage detection framework for multi-oriented scene text. It employs corners to estimate the possible locations of text instances and a region-wise subnetwork for further classification and regression. In our experiments, it achieves F-measure of 0.876 and 0.845 on ICDAR 2013 and 2015 respectively. The paper is available at [arXiv](https://arxiv.org/abs/1804.02690).


### Installation

This code is based on [Caffe](https://github.com/BVLC/caffe) and [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn). It has been tested on Ubuntu 16.04 with CUDA 8.0.

0. Clone this repository
    ```
    git clone https://github.com/xhzdeng/crpn.git
    ```

1. Build Caffe and pycaffe 
    ```
    cd $CRPN_ROOT/caffe-fast-rcnn
    make -j8 && make pycaffe
    ```

2. Build the Cython modules
    ```
    cd $CRPN_ROOT/lib
    make
    ```

3. Prepare your own training data directory. For convenience, it should have this basic structure.
	```
	$VOCdevkit/
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc. 
    ```
   And create symlinks for YOUR dataset
    ```
    cd $CRPN_ROOT/data
    ln -s [path] VOCdevkit
    ```

4. Download pretrained ImageNet VGG-16 model. You can find it at [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo).

5. Train with YOUR dataset
    ```
    cd $CRPN_ROOT
    ./experiments/scripts/train.sh [NET] [MODEL] [DATASET] [ITER_NUM]
    # NET is the network arch to use, only {vgg16} in this implemention
    # MODEL is the pre-trained model you want to use to initial your weights
    # DATASET points to your dataset, please refer the contents of train.sh
    # IETR_NUM 
    ```

6. Test with YOUR models
    ```
    cd $CRPN_ROOT
    ./experiments/scripts/test.sh [NET] [MODEL] [DATASET]
    # NET is the network arch to use, only {vgg16} in this implemention
    # MODEL is the testing model
    # DATASET points to your dataset, please refer the contents of test.sh
    ```
    Test outputs are saved under:
    ```
    output/<experiment directory>/<dataset name>/<network snapshot name>/
    ```



### Models

Now, you can download the pretrained model from [OneDrive](https://1drv.ms/f/s!AiAzf2_GWxxlefnWI2-umwO3R9g) or [BaiduYun](https://pan.baidu.com/s/1Ivk4v49w0oW4VzWQMMEqcQ), which is trained 100k iters on [SynthText](https://github.com/ankush-me/SynthText). I also have uploaded a testing model trained recently. It achieves an F-measure of 0.8456 at 840p resolution on ICDAR 2015, similar performance but slightly faster than we depicted in the paper.


### Citation

If you find the paper and code useful in your research, please consider citing:

    @article{deng2018crpn,
        Title = {Detecting Multi-Oriented Text with Corner-based Region Proposals},
        Author = {Linjie Deng and Yanxiang Gong and Yi Lin and Jingwen Shuai and Xiaoguang Tu and Yufei Zhang and Zheng Ma and Mei Xie},
        Journal = {arXiv preprint arXiv:1804.02690},
        Year = {2018}
    }
















