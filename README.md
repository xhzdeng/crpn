# Corner-based Region Proposal Network

CRPN is a two-stage detection framework for multi-oriented scene text. It employs corners to estimate the possible locations of text instances and a region-wise subnetwork for further classification and regression. In our experiments, it achieves F-measure of 0.876 and 0.845 on ICDAR 2013 and 2015 respectively. The paper will be released soon. 


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

4. Download pre-trained ImageNet VGG-16 models. You can find it at [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)

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

