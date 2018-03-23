# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from fast_rcnn.config import cfg
from nms.gpu_nms import gpu_nms
from nms_quad.gpu_nms import gpu_nms_quad


def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS and not force_cpu:
        if dets.shape[1] == 9:
            return gpu_nms_quad(dets, thresh, device_id=cfg.GPU_ID)
        else:
            return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
