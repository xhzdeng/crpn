# Corner-based Region Proposal Network

CRPN is a two-stage detection framework for multi-oriented scene text. The code is modified from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn). 


### Requirements

0. Clone this repository
    ```
    git clone https://github.com/xhzdeng/crpn.git
    ```

1. Build Caffe and pycaffe (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
    ```
    cd $CRPN_ROOT/caffe-fast-rcnn
    make -j8 && make pycaffe
    ```

2. Build the Cython modules
    ```
    cd $CRPN_ROOT/lib
    make
    ```

3. Prepare your own training data directory. It should have this basic structure.
	```
	$VOCdevkit/                           # development kit
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
    # DATASET points to your dataset, please refer the train.sh file
    # IETR_NUM 
    ```

6. Test with YOUR models
    ```
    cd $CRPN_ROOT
    ./experiments/scripts/test.sh [NET] [MODEL] [DATASET]
    # NET is the network arch to use, only {vgg16} in this implemention
    # MODEL is the testing model
    # DATASET points to your dataset, please refer the test.sh file
    ```
    Test outputs are saved under:
    ```
    output/<experiment directory>/<dataset name>/<network snapshot name>/
    ```

