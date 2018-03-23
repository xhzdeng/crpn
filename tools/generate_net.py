#!/usr/bin/env python

from __future__ import division
import _init_paths
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop


def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)


def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def network(split):

    n = caffe.NetSpec()

    if split == 'train':
        pymodule = 'roi_data_layer.layer'
        pylayer = 'RoIDataLayer'
        pydata_params = dict(num_classes=2)
        n.data, n.im_info, n.gt_boxes = L.Python(module=pymodule, layer=pylayer, ntop=3, param_str=str(pydata_params))
    else:
        n.data = L.Input(name='data', input_param=dict(shape=dict(dim=[1, 3, 512, 512])))
        n.im_info = L.Input(name='im_info', input_param=dict(shape=dict(dim=[1, 3])))

    # the base net: vgg16
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=1)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)
    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)
    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)
    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)
    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    # n.pool5 = max_pool(n.relu5_3)

    # FEATURE MAP #
    # n.conv_rpn = L.Convolution(
    #     n.conv5_3, num_output=256, kernel_size=1, pad=0,
    #     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
    #     weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    # n.conv_rpn_relu = L.ReLU(n.conv_rpn, in_place=True)

    # reduct dims
    n.conv5_4 = L.Convolution(
        n.conv5_3, num_output=256, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    n.conv5_4_relu = L.ReLU(n.conv5_4, in_place=True)
    # upsample reference from RON
    n.upsample5 = L.Deconvolution(
        n.conv5_4, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        convolution_param=dict(num_output=256, kernel_size=2, stride=2,
                               weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')))
    n.upsample5_relu = L.ReLU(n.upsample5, in_place=True)
    # reduct dims
    n.conv4_4 = L.Convolution(
        n.conv4_3, num_output=256, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    n.conv4_4_relu = L.ReLU(n.conv4_4, in_place=True)
    # concat
    n.concat = L.Concat(n.upsample5, n.conv4_4, name='concat', concat_param=dict({'concat_dim': 1}))
    n.conv_hyper = L.Convolution(
        n.concat, num_output=256, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    n.conv_hyper_relu = L.ReLU(n.conv_hyper, in_place=True)
    # conv
    n.conv_rpn = L.Convolution(
        n.conv_hyper, num_output=256, kernel_size=3, pad=1,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    n.conv_rpn_relu = L.ReLU(n.conv_rpn, in_place=True)

    # CROSS ENTROPY VERSION #
    # top_left
    n.rpn_score_tl = L.Convolution(
        n.conv_rpn, num_output=1, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    n.rpn_score_tr = L.Convolution(
        n.conv_rpn, num_output=1, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    n.rpn_score_br = L.Convolution(
        n.conv_rpn, num_output=1, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    n.rpn_score_bl = L.Convolution(
        n.conv_rpn, num_output=1, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))

    if split == 'train':
        pymodule = 'rpn.cornermap_layer'
        pylayer = 'CornerMapLayer'
        n.rpn_map_tl, n.rpn_map_tr, n.rpn_map_br, n.rpn_map_bl = \
            L.Python(n.conv_rpn, n.im_info, n.gt_boxes, module=pymodule, layer=pylayer, ntop=4)
        n.loss_rpn_tl = L.SigmoidCrossEntropyLoss(
            n.rpn_score_tl, n.rpn_map_tl, propagate_down=[1, 0],
            loss_param=dict(normalize=True, ignore_label=-1))
        n.loss_rpn_tr = L.SigmoidCrossEntropyLoss(
            n.rpn_score_tr, n.rpn_map_tr, propagate_down=[1, 0],
            loss_param=dict(normalize=True, ignore_label=-1))
        n.loss_rpn_br = L.SigmoidCrossEntropyLoss(
            n.rpn_score_br, n.rpn_map_br, propagate_down=[1, 0],
            loss_param=dict(normalize=True, ignore_label=-1))
        n.loss_rpn_bl = L.SigmoidCrossEntropyLoss(
            n.rpn_score_bl, n.rpn_map_bl, propagate_down=[1, 0],
            loss_param=dict(normalize=True, ignore_label=-1))

    if split == 'test':
        n.rpn_prob_tl = L.Sigmoid(n.rpn_score_tl)
        n.rpn_prob_tr = L.Sigmoid(n.rpn_score_tr)
        n.rpn_prob_br = L.Sigmoid(n.rpn_score_br)
        n.rpn_prob_bl = L.Sigmoid(n.rpn_score_bl)
        pymodule = 'rpn.quad_layer'
        pylayer = 'QuadLayer'
        n.quads, n.rois, n.cls_prob = L.Python(n.rpn_prob_tl, n.rpn_prob_tr, n.rpn_prob_br, n.rpn_prob_bl,
                                               module=pymodule, layer=pylayer, ntop=3)
    # CROSS ENTROPY VERSION #

    return n.to_proto()


def make_net():
    with open('train.pt', 'w') as f:
        f.write(str(network('train')))
    with open('test.pt', 'w') as f:
        f.write(str(network('test')))


if __name__ == '__main__':
    make_net()
