# --------------------------------------------------------
# CRPN
# Written by Linjie Deng
# --------------------------------------------------------
import caffe
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg


DEBUG = False


class LabelMapLayer(caffe.Layer):

    def setup(self, bottom, top):
        # no extra param
        height, width = bottom[0].data.shape[-2:]
        top[0].reshape(1, 1, height, width)
        top[1].reshape(1, 1, height, width)
        top[2].reshape(1, 1, height, width)
        top[3].reshape(1, 1, height, width)

    def forward(self, bottom, top):
        # params
        batch_size = 32
        fg_fraction = 1.0
        num_fg = int(fg_fraction * batch_size)
        theta_interval = cfg.LD_INTERVAL

        feat_map = bottom[0].data
        im_info = bottom[1].data[0, :]
        gt_boxes = bottom[2].data
        img_h, img_w = im_info[:2]
        map_h, map_w = feat_map.shape[-2:]

        spatial_scale_x = map_w / img_w
        spatial_scale_y = map_h / img_h
        assert spatial_scale_x == spatial_scale_y, 'scale_x is not equal to scale_y'
        spatial_scale = spatial_scale_x

        labelmap_tl = np.zeros((map_h, map_w), dtype=np.float32)
        labelmap_tr = np.zeros((map_h, map_w), dtype=np.float32)
        labelmap_br = np.zeros((map_h, map_w), dtype=np.float32)
        labelmap_bl = np.zeros((map_h, map_w), dtype=np.float32)

        for bbox in gt_boxes:
            x1_valid = (0 <= bbox[0] < img_w)
            y1_valid = (0 <= bbox[1] < img_h)
            x2_valid = (0 <= bbox[2] < img_w)
            y2_valid = (0 <= bbox[3] < img_h)
            x3_valid = (0 <= bbox[4] < img_w)
            y3_valid = (0 <= bbox[5] < img_h)
            x4_valid = (0 <= bbox[6] < img_w)
            y4_valid = (0 <= bbox[7] < img_h)

            x1 = int(round(bbox[0] * spatial_scale))
            y1 = int(round(bbox[1] * spatial_scale))
            x2 = int(round(bbox[2] * spatial_scale))
            y2 = int(round(bbox[3] * spatial_scale))
            x3 = int(round(bbox[4] * spatial_scale))
            y3 = int(round(bbox[5] * spatial_scale))
            x4 = int(round(bbox[6] * spatial_scale))
            y4 = int(round(bbox[7] * spatial_scale))
            x1 = np.maximum(np.minimum(x1, map_w - 1), 0)
            y1 = np.maximum(np.minimum(y1, map_h - 1), 0)
            x2 = np.maximum(np.minimum(x2, map_w - 1), 0)
            y2 = np.maximum(np.minimum(y2, map_h - 1), 0)
            x3 = np.maximum(np.minimum(x3, map_w - 1), 0)
            y3 = np.maximum(np.minimum(y3, map_h - 1), 0)
            x4 = np.maximum(np.minimum(x4, map_w - 1), 0)
            y4 = np.maximum(np.minimum(y4, map_h - 1), 0)

            # rescaled or unrescaled?
            # theta1 = _compute_theta(bbox[0], bbox[1], bbox[4], bbox[5])
            # theta2 = _compute_theta(bbox[2], bbox[3], bbox[6], bbox[7])
            # theta3 = _compute_theta(bbox[4], bbox[5], bbox[0], bbox[1])
            # theta4 = _compute_theta(bbox[6], bbox[7], bbox[2], bbox[3])
            theta1 = _compute_theta(x1, y1, x3, y3)
            theta2 = _compute_theta(x2, y2, x4, y4)
            theta3 = _compute_theta(x3, y3, x1, y1)
            theta4 = _compute_theta(x4, y4, x2, y2)

            # positive sample's index start from 1
            if x1_valid and y1_valid:
                labelmap_tl[y1, x1] = np.floor(theta1 / theta_interval) + 1
            if x2_valid and y2_valid:
                labelmap_tr[y2, x2] = np.floor(theta2 / theta_interval) + 1
            if x3_valid and y3_valid:
                labelmap_br[y3, x3] = np.floor(theta3 / theta_interval) + 1
            if x4_valid and y4_valid:
                labelmap_bl[y4, x4] = np.floor(theta4 / theta_interval) + 1

        # subsample positive or negative labels if we have too many
        labelmap_tl = _subsample(labelmap_tl, batch_size, num_fg)
        labelmap_tr = _subsample(labelmap_tr, batch_size, num_fg)
        labelmap_br = _subsample(labelmap_br, batch_size, num_fg)
        labelmap_bl = _subsample(labelmap_bl, batch_size, num_fg)

        labelmap_tl = labelmap_tl.reshape((1, map_h, map_w, 1)).transpose(0, 3, 1, 2)
        labelmap_tl = labelmap_tl.reshape((1, 1, map_h, map_w))
        top[0].reshape(*labelmap_tl.shape)
        top[0].data[...] = labelmap_tl

        labelmap_tr = labelmap_tr.reshape((1, map_h, map_w, 1)).transpose(0, 3, 1, 2)
        labelmap_tr = labelmap_tr.reshape((1, 1, map_h, map_w))
        top[1].reshape(*labelmap_tr.shape)
        top[1].data[...] = labelmap_tr

        labelmap_br = labelmap_br.reshape((1, map_h, map_w, 1)).transpose(0, 3, 1, 2)
        labelmap_br = labelmap_br.reshape((1, 1, map_h, map_w))
        top[2].reshape(*labelmap_br.shape)
        top[2].data[...] = labelmap_br

        labelmap_bl = labelmap_bl.reshape((1, map_h, map_w, 1)).transpose(0, 3, 1, 2)
        labelmap_bl = labelmap_bl.reshape((1, 1, map_h, map_w))
        top[3].reshape(*labelmap_bl.shape)
        top[3].data[...] = labelmap_bl

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _subsample(labelmap, batch_size, num_fg):
    # positive
    labelmap = labelmap.reshape(-1, 1)
    fg_inds = np.where(labelmap > 0)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labelmap[disable_inds] = -1
    # negative
    num_bg = batch_size - np.sum(labelmap > 0)
    bg_inds = np.where(labelmap == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labelmap[disable_inds] = -1
    return labelmap


def _compute_theta(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    val = dx / np.sqrt(dx * dx + dy * dy)
    val = np.maximum(np.minimum(val, 1), -1)
    theta = np.arccos(val) / np.pi * 180
    if dy > 0:
        theta = 360 - theta
    return theta