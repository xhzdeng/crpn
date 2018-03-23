#!/usr/bin/env python
from __future__ import division
import _init_paths
import caffe
from caffe import layers as L, params as P
from fast_rcnn.config import cfg


def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)


def conv_relu_fix(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad,
                         param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)


def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def network(split):

    num_chns = int(360 / cfg.LD_INTERVAL) + 1
    net = caffe.NetSpec()

    if split == 'train':
        pymodule = 'roi_data_layer.layer'
        pylayer = 'RoIDataLayer'
        pydata_params = dict(num_classes=2)
        net.data, net.im_info, net.gt_boxes = L.Python(
            module=pymodule, layer=pylayer, ntop=3, param_str=str(pydata_params))
    else:
        net.data = L.Input(name='data', input_param=dict(shape=dict(dim=[1, 3, 512, 512])))
        net.im_info = L.Input(name='im_info', input_param=dict(shape=dict(dim=[1, 3])))

    # Backbone
    net.conv1_1, net.relu1_1 = conv_relu(net.data, 64, pad=1)
    net.conv1_2, net.relu1_2 = conv_relu(net.relu1_1, 64)
    net.pool1 = max_pool(net.relu1_2)
    net.conv2_1, net.relu2_1 = conv_relu(net.pool1, 128)
    net.conv2_2, net.relu2_2 = conv_relu(net.relu2_1, 128)
    net.pool2 = max_pool(net.relu2_2)
    net.conv3_1, net.relu3_1 = conv_relu(net.pool2, 256)
    net.conv3_2, net.relu3_2 = conv_relu(net.relu3_1, 256)
    net.conv3_3, net.relu3_3 = conv_relu(net.relu3_2, 256)
    net.pool3 = max_pool(net.relu3_3)
    net.conv4_1, net.relu4_1 = conv_relu(net.pool3, 512)
    net.conv4_2, net.relu4_2 = conv_relu(net.relu4_1, 512)
    net.conv4_3, net.relu4_3 = conv_relu(net.relu4_2, 512)
    net.pool4 = max_pool(net.relu4_3)
    net.conv5_1, net.relu5_1 = conv_relu(net.pool4, 512)
    net.conv5_2, net.relu5_2 = conv_relu(net.relu5_1, 512)
    net.conv5_3, net.relu5_3 = conv_relu(net.relu5_2, 512)
    # net.pool_5 = max_pool(net.relu5_3)

    # Hyper Feature
    net.downsample = L.Convolution(
        net.conv3_3, num_output=64, kernel_size=3, pad=1, stride=2,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    net.relu_downsample = L.ReLU(net.downsample, in_place=True)
    net.upsample = L.Deconvolution(
        net.conv5_3, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        convolution_param=dict(num_output=512, kernel_size=2, stride=2,
        weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')))
    net.relu_upsample = L.ReLU(net.upsample, in_place=True)
    net.fuse = L.Concat(net.downsample, net.upsample, net.conv4_3, name='concat', concat_param=dict({'concat_dim': 1}))
    net.conv_hyper = L.Convolution(
        net.fuse, num_output=512, kernel_size=3, pad=1,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    net.relu_conv_hyper = L.ReLU(net.conv_hyper, in_place=True)

    net.conv_rpn = L.Convolution(
        net.conv_hyper, num_output=128, kernel_size=3, pad=1,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    net.conv_rpn_relu = L.ReLU(net.conv_rpn, in_place=True)
    net.rpn_score_tl = L.Convolution(
        net.conv_rpn, num_output=num_chns, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    net.rpn_score_tr = L.Convolution(
        net.conv_rpn, num_output=num_chns, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    net.rpn_score_br = L.Convolution(
        net.conv_rpn, num_output=num_chns, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    net.rpn_score_bl = L.Convolution(
        net.conv_rpn, num_output=num_chns, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    net.rpn_prob_tl = L.Softmax(net.rpn_score_tl)
    net.rpn_prob_tr = L.Softmax(net.rpn_score_tr)
    net.rpn_prob_br = L.Softmax(net.rpn_score_br)
    net.rpn_prob_bl = L.Softmax(net.rpn_score_bl)

    if split == 'train':
        pymodule = 'rpn.labelmap_layer'
        pylayer = 'LabelMapLayer'
        net.rpn_label_tl, net.rpn_label_tr, net.rpn_label_br, net.rpn_label_bl = L.Python(
            net.conv_rpn, net.im_info, net.gt_boxes, module=pymodule, layer=pylayer, ntop=4)
        net.loss_rpn_tl = L.BalancedSoftmaxWithLoss(
            net.rpn_score_tl, net.rpn_label_tl, propagate_down=[1, 0],
            loss_param=dict(normalize=True, ignore_label=-1))
        net.loss_rpn_tr = L.BalancedSoftmaxWithLoss(
            net.rpn_score_tr, net.rpn_label_tr, propagate_down=[1, 0],
            loss_param=dict(normalize=True, ignore_label=-1))
        net.loss_rpn_br = L.BalancedSoftmaxWithLoss(
            net.rpn_score_br, net.rpn_label_br, propagate_down=[1, 0],
            loss_param=dict(normalize=True, ignore_label=-1))
        net.loss_rpn_bl = L.BalancedSoftmaxWithLoss(
            net.rpn_score_bl, net.rpn_label_bl, propagate_down=[1, 0],
            loss_param=dict(normalize=True, ignore_label=-1))
        pymodule = 'rpn.proposal_layer'
        pylayer = 'ProposalLayer'
        pydata_params = dict(feat_stride=8)
        net.quads = L.Python(
            net.im_info, net.rpn_prob_tl, net.rpn_prob_tr, net.rpn_prob_br, net.rpn_prob_bl,
            module=pymodule, layer=pylayer, ntop=1, param_str=str(pydata_params))
        pymodule = 'rpn.proposal_target_layer'
        pylayer = 'ProposalTargetLayer'
        net.rois, net.labels, net.bbox_targets, net.bbox_inside_weights, net.bbox_outside_weights = L.Python(
            net.quads, net.gt_boxes, module=pymodule, layer=pylayer, name='roi-data', ntop=5)
        # RCNN
        net.dual_pool5 = L.RotateROIPooling(
            net.conv_hyper, net.rois, name='roi_pool5_dual',
            rotate_roi_pooling_param=dict(pooled_w=7, pooled_h=7, spatial_scale=0.125))
        net.pool5_a, net.pool5_b = L.Slice(net.dual_pool5, slice_param=dict(axis=0), ntop=2, name='slice')
        net.pool5 = L.Eltwise(net.pool5_a, net.pool5_b, name='roi_pool5', eltwise_param=dict(operation=1))
        net.fc6 = L.InnerProduct(
            net.pool5, param=[dict(lr_mult=1), dict(lr_mult=2)],
            inner_product_param=dict(num_output=4096,
            weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')))
        net.fc6_relu = L.ReLU(net.fc6, in_place=True)
        net.drop6 = L.Dropout(net.fc6, dropout_ratio=0.5, in_place=True)
        net.fc7 = L.InnerProduct(
            net.fc6, param=[dict(lr_mult=1), dict(lr_mult=2)],
            inner_product_param=dict(num_output=4096,
            weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')))
        net.fc7_relu = L.ReLU(net.fc7, in_place=True)
        net.drop7 = L.Dropout(net.fc7, dropout_ratio=0.5, in_place=True)
        net.cls_score = L.InnerProduct(
            net.fc7, param=[dict(lr_mult=1), dict(lr_mult=2)],
            inner_product_param=dict(num_output=2,
            weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0)))
        net.bbox_pred = L.InnerProduct(
            net.fc7, param=[dict(lr_mult=1), dict(lr_mult=2)],
            inner_product_param=dict(num_output=16,
            weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0)))
        net.loss_cls = L.SoftmaxWithLoss(net.cls_score, net.labels, propagate_down=[1, 0], loss_weight=1)
        net.loss_bbox = L.SmoothL1Loss(net.bbox_pred, net.bbox_targets, net.bbox_inside_weights,
                                       net.bbox_outside_weights, loss_weight=1)

    if split == 'test':
        pymodule = 'rpn.proposal_layer'
        pylayer = 'ProposalLayer'
        pydata_params = dict(feat_stride=8)
        net.quads, net.rois = L.Python(
            net.im_info, net.rpn_prob_tl, net.rpn_prob_tr, net.rpn_prob_br, net.rpn_prob_bl,
            module=pymodule, layer=pylayer, ntop=2, param_str=str(pydata_params))
        # RCNN
        net.dual_pool5 = L.RotateROIPooling(
            net.conv_hyper, net.rois, name='roi_pool5_dual',
            rotate_roi_pooling_param=dict(pooled_w=7, pooled_h=7, spatial_scale=0.125))
        net.pool5_a, net.pool5_b = L.Slice(net.dual_pool5, slice_param=dict(axis=0), ntop=2, name='slice')
        net.pool5 = L.Eltwise(net.pool5_a, net.pool5_b, name='roi_pool5', eltwise_param=dict(operation=1))
        net.fc6 = L.InnerProduct(
            net.pool5, param=[dict(lr_mult=1), dict(lr_mult=2)],
            inner_product_param=dict(num_output=4096,
            weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')))
        net.fc6_relu = L.ReLU(net.fc6, in_place=True)
        net.drop6 = L.Dropout(net.fc6, dropout_ratio=0.5, in_place=True)
        net.fc7 = L.InnerProduct(
            net.fc6, param=[dict(lr_mult=1), dict(lr_mult=2)],
            inner_product_param=dict(num_output=4096,
            weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')))
        net.fc7_relu = L.ReLU(net.fc7, in_place=True)
        net.drop7 = L.Dropout(net.fc7, dropout_ratio=0.5, in_place=True)
        net.cls_score = L.InnerProduct(
            net.fc7, param=[dict(lr_mult=1), dict(lr_mult=2)],
            inner_product_param=dict(num_output=2,
            weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0)))
        net.bbox_pred = L.InnerProduct(
            net.fc7, param=[dict(lr_mult=1), dict(lr_mult=2)],
            inner_product_param=dict(num_output=16,
            weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0)))
        net.cls_prob = L.Softmax(net.cls_score)

    return net.to_proto()


def make_net():
    with open('train.pt', 'w') as f:
        f.write(str(network('train')))
    with open('test.pt', 'w') as f:
        f.write(str(network('test')))


if __name__ == '__main__':
    make_net()
