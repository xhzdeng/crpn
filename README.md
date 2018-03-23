# Corner-based Region Proposal Network

CRPN is a two-stage detection framework for multi-oriented scene text. The code is modified from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn). 


### Requirements

0. Clone the CRPN repository
    ```
    git clone https://github.com/xhzdeng/crpn.git
    ```

1. Build Caffe and pycaffe (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
    ```
    cd caffe-fast-rcnn
    make -j8 && make pycaffe
    ```

2. Build the Cython modules
    ```
    cd $CRPN_ROOT/lib
    make
    ```

3. Prepare your own training data. It should have the basic structure followed PASCAL VOC dataset
    ```
    Create symlinks for YOUR dataset
    cd $CRPN_ROOT/data
    ln -s [dataset_path] VOCdevkit
    ```

4. Download pre-trained ImageNet VGG-16 models
    ```
    You can find it at [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
    ```
5. Train with YOUR dataset
    ```
    cd $CRPN_ROOT
    ./experiments/scripts/train.sh [NET] [MODEL] [DATASET] [ITER_NUM]
    # NET is the network arch to use, only {vgg16} in this implemention
    # MODEL is the pre-trained model you want to use to initial your weights
    # DATASET points to your dataset
    # IETR_NUM 
    ```

6. Test with YOUR dataset
    ```
    cd $CRPN_ROOT
    ./experiments/scripts/test.sh [NET] [MODEL] [DATASET]
    # NET is the network arch to use, only {vgg16} in this implemention
    # MODEL is the resulting model you trained before
    # DATASET points to your dataset
    ```
    Test outputs are saved under:
    ```
    output/<experiment directory>/<dataset name>/<network snapshot name>/
    ```

