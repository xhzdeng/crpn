# --------------------------------------------------------
# CRPN
# Written by Linjie Deng
# --------------------------------------------------------
import numpy as np
from quad_convert import quad_2_aabb
from sort_points import sort_points


def quad_transform(ex_rois, gt_rois):

    ex_rois = sort_points(ex_rois)

    ex_aabbs = quad_2_aabb(ex_rois)
    ex_widths = ex_aabbs[:, 2] - ex_aabbs[:, 0] + 1.0
    ex_heights = ex_aabbs[:, 3] - ex_aabbs[:, 1] + 1.0

    # REFEFENCE FROM [2017 CVPR] DMPNet
    # ex_ctr_x = ex_aabbs[:, 0] + 0.5 * ex_widths
    # ex_ctr_y = ex_aabbs[:, 1] + 0.5 * ex_heights
    # ex_w1 = ex_rois[:, 0] - ex_ctr_x
    # ex_h1 = ex_rois[:, 1] - ex_ctr_y
    # ex_w2 = ex_rois[:, 2] - ex_ctr_x
    # ex_h2 = ex_rois[:, 3] - ex_ctr_y
    # ex_w3 = ex_rois[:, 4] - ex_ctr_x
    # ex_h3 = ex_rois[:, 5] - ex_ctr_y
    # ex_w4 = ex_rois[:, 6] - ex_ctr_x
    # ex_h4 = ex_rois[:, 7] - ex_ctr_y
    #
    # gt_aabbs = quad_2_aabb(gt_rois)
    # gt_widths = gt_aabbs[:, 2] - gt_aabbs[:, 0] + 1.0
    # gt_heights = gt_aabbs[:, 3] - gt_aabbs[:, 1] + 1.0
    # gt_ctr_x = gt_aabbs[:, 0] + 0.5 * gt_widths
    # gt_ctr_y = gt_aabbs[:, 1] + 0.5 * gt_heights
    # gt_w1 = gt_rois[:, 0] - gt_ctr_x
    # gt_h1 = gt_rois[:, 1] - gt_ctr_y
    # gt_w2 = gt_rois[:, 2] - gt_ctr_x
    # gt_h2 = gt_rois[:, 3] - gt_ctr_y
    # gt_w3 = gt_rois[:, 4] - gt_ctr_x
    # gt_h3 = gt_rois[:, 5] - gt_ctr_y
    # gt_w4 = gt_rois[:, 6] - gt_ctr_x
    # gt_h4 = gt_rois[:, 7] - gt_ctr_y
    #
    # targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    # targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    # targets_w1 = (gt_w1 - ex_w1) / ex_widths
    # targets_h1 = (gt_h1 - ex_h1) / ex_heights
    # targets_w2 = (gt_w2 - ex_w2) / ex_widths
    # targets_h2 = (gt_h2 - ex_h2) / ex_heights
    # targets_w3 = (gt_w3 - ex_w3) / ex_widths
    # targets_h3 = (gt_h3 - ex_h3) / ex_heights
    # targets_w4 = (gt_w4 - ex_w4) / ex_widths
    # targets_h4 = (gt_h4 - ex_h4) / ex_heights
    # targets = np.vstack(
    #     (targets_dx, targets_dy, targets_w1, targets_h1,
    #      targets_w2, targets_h2, targets_w3, targets_h3, targets_w4, targets_h4)).transpose()

    # SIMPLE ONE
    ex_x1 = ex_rois[:, 0]
    ex_y1 = ex_rois[:, 1]
    ex_x2 = ex_rois[:, 2]
    ex_y2 = ex_rois[:, 3]
    ex_x3 = ex_rois[:, 4]
    ex_y3 = ex_rois[:, 5]
    ex_x4 = ex_rois[:, 6]
    ex_y4 = ex_rois[:, 7]

    gt_x1 = gt_rois[:, 0]
    gt_y1 = gt_rois[:, 1]
    gt_x2 = gt_rois[:, 2]
    gt_y2 = gt_rois[:, 3]
    gt_x3 = gt_rois[:, 4]
    gt_y3 = gt_rois[:, 5]
    gt_x4 = gt_rois[:, 6]
    gt_y4 = gt_rois[:, 7]

    target_dx1 = (gt_x1 - ex_x1) / ex_widths
    target_dy1 = (gt_y1 - ex_y1) / ex_heights
    target_dx2 = (gt_x2 - ex_x2) / ex_widths
    target_dy2 = (gt_y2 - ex_y2) / ex_heights
    target_dx3 = (gt_x3 - ex_x3) / ex_widths
    target_dy3 = (gt_y3 - ex_y3) / ex_heights
    target_dx4 = (gt_x4 - ex_x4) / ex_widths
    target_dy4 = (gt_y4 - ex_y4) / ex_heights

    targets = np.vstack(
        (target_dx1, target_dy1, target_dx2, target_dy2,
         target_dx3, target_dy3, target_dx4, target_dy4)).transpose()

    return targets


def quad_transform_inv(quads, deltas):

    if quads.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    quads = sort_points(quads)

    # REFEFENCE FROM [2017 CVPR] DMPNet
    # aabbs = quad_2_aabb(quads)
    # widths = aabbs[:, 2] - aabbs[:, 0] + 1.0
    # heights = aabbs[:, 3] - aabbs[:, 1] + 1.0
    # ctr_x = aabbs[:, 0] + 0.5 * widths
    # ctr_y = aabbs[:, 1] + 0.5 * heights
    #
    # w1 = quads[:, 0] - ctr_x
    # h1 = quads[:, 1] - ctr_y
    # w2 = quads[:, 2] - ctr_x
    # h2 = quads[:, 3] - ctr_y
    # w3 = quads[:, 4] - ctr_x
    # h3 = quads[:, 5] - ctr_y
    # w4 = quads[:, 6] - ctr_x
    # h4 = quads[:, 7] - ctr_y
    #
    # dx = deltas[:, 0::10]
    # dy = deltas[:, 1::10]
    # dw1 = deltas[:, 2::10]
    # dh1 = deltas[:, 3::10]
    # dw2 = deltas[:, 4::10]
    # dh2 = deltas[:, 5::10]
    # dw3 = deltas[:, 6::10]
    # dh3 = deltas[:, 7::10]
    # dw4 = deltas[:, 8::10]
    # dh4 = deltas[:, 9::10]
    #
    # pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    # pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    # pred_w1 = dw1 * widths[:, np.newaxis] + w1[:, np.newaxis]
    # pred_h1 = dh1 * heights[:, np.newaxis] + h1[:, np.newaxis]
    # pred_w2 = dw2 * widths[:, np.newaxis] + w2[:, np.newaxis]
    # pred_h2 = dh2 * heights[:, np.newaxis] + h2[:, np.newaxis]
    # pred_w3 = dw3 * widths[:, np.newaxis] + w3[:, np.newaxis]
    # pred_h3 = dh3 * heights[:, np.newaxis] + h3[:, np.newaxis]
    # pred_w4 = dw4 * widths[:, np.newaxis] + w4[:, np.newaxis]
    # pred_h4 = dh4 * heights[:, np.newaxis] + h4[:, np.newaxis]
    #
    # pred_boxes = np.zeros([quads.shape[0], 16], dtype=deltas.dtype)
    # pred_boxes[:, 0::8] = pred_ctr_x + pred_w1
    # pred_boxes[:, 1::8] = pred_ctr_y + pred_h1
    # pred_boxes[:, 2::8] = pred_ctr_x + pred_w2
    # pred_boxes[:, 3::8] = pred_ctr_y + pred_h2
    # pred_boxes[:, 4::8] = pred_ctr_x + pred_w3
    # pred_boxes[:, 5::8] = pred_ctr_y + pred_h3
    # pred_boxes[:, 6::8] = pred_ctr_x + pred_w4
    # pred_boxes[:, 7::8] = pred_ctr_y + pred_h4

    aabbs = quad_2_aabb(quads)
    widths = aabbs[:, 2] - aabbs[:, 0] + 1.0
    heights = aabbs[:, 3] - aabbs[:, 1] + 1.0
    x1 = quads[:, 0]
    y1 = quads[:, 1]
    x2 = quads[:, 2]
    y2 = quads[:, 3]
    x3 = quads[:, 4]
    y3 = quads[:, 5]
    x4 = quads[:, 6]
    y4 = quads[:, 7]

    dx1 = deltas[:, 0::8]
    dy1 = deltas[:, 1::8]
    dx2 = deltas[:, 2::8]
    dy2 = deltas[:, 3::8]
    dx3 = deltas[:, 4::8]
    dy3 = deltas[:, 5::8]
    dx4 = deltas[:, 6::8]
    dy4 = deltas[:, 7::8]

    pred_x1 = dx1 * widths[:, np.newaxis] + x1[:, np.newaxis]
    pred_y1 = dy1 * heights[:, np.newaxis] + y1[:, np.newaxis]
    pred_x2 = dx2 * widths[:, np.newaxis] + x2[:, np.newaxis]
    pred_y2 = dy2 * heights[:, np.newaxis] + y2[:, np.newaxis]
    pred_x3 = dx3 * widths[:, np.newaxis] + x3[:, np.newaxis]
    pred_y3 = dy3 * heights[:, np.newaxis] + y3[:, np.newaxis]
    pred_x4 = dx4 * widths[:, np.newaxis] + x4[:, np.newaxis]
    pred_y4 = dy4 * heights[:, np.newaxis] + y4[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    pred_boxes[:, 0::8] = pred_x1
    pred_boxes[:, 1::8] = pred_y1
    pred_boxes[:, 2::8] = pred_x2
    pred_boxes[:, 3::8] = pred_y2
    pred_boxes[:, 4::8] = pred_x3
    pred_boxes[:, 5::8] = pred_y3
    pred_boxes[:, 6::8] = pred_x4
    pred_boxes[:, 7::8] = pred_y4

    return pred_boxes


def clip_quads(quads, im_shape):
    """
    Clip quads to image boundaries.
    """
    quads[:, 0::8] = np.maximum(np.minimum(quads[:, 0::8], im_shape[1] - 1), 0)
    quads[:, 1::8] = np.maximum(np.minimum(quads[:, 1::8], im_shape[0] - 1), 0)
    quads[:, 2::8] = np.maximum(np.minimum(quads[:, 2::8], im_shape[1] - 1), 0)
    quads[:, 3::8] = np.maximum(np.minimum(quads[:, 3::8], im_shape[0] - 1), 0)
    quads[:, 4::8] = np.maximum(np.minimum(quads[:, 4::8], im_shape[1] - 1), 0)
    quads[:, 5::8] = np.maximum(np.minimum(quads[:, 5::8], im_shape[0] - 1), 0)
    quads[:, 6::8] = np.maximum(np.minimum(quads[:, 6::8], im_shape[1] - 1), 0)
    quads[:, 7::8] = np.maximum(np.minimum(quads[:, 7::8], im_shape[0] - 1), 0)
    return quads
