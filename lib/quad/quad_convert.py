# --------------------------------------------------------
# CRPN
# Written by Linjie Deng
# --------------------------------------------------------

import numpy as np
import math

# aabb: [xmin, ymin, xmax, ymax]
# obb:  [xmin, ymin, xmax, ymax, theta]
# quad: [x1, y1, x2, y2, x3, y3, x4, y4]


def whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[:, 2] - anchor[:, 0] + 1
    h = anchor[:, 3] - anchor[:, 1] + 1
    x_ctr = anchor[:, 0] + 0.5 * (w - 1)
    y_ctr = anchor[:, 1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    anchors = np.vstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1))).astype(np.float32, copy=False)
    return anchors.transpose()


def dilate_roi(rois, scale=1.2):
    ws, hs, x_ctr, y_ctr = whctrs(rois[:, :4])
    new_ws = ws * scale
    new_hs = hs * scale
    rois[:, :4] = mkanchors(new_ws, new_hs, x_ctr, y_ctr)
    return rois


def dual_roi(rois):
    # O_1 = [x, y, w, h, t]
    # O_2 = [x, y, h, w, t - 90]
    temp = np.zeros(rois.shape)
    ws, hs, x_ctr, y_ctr = whctrs(rois[:, :4])
    temp[:, :4] = mkanchors(hs, ws, x_ctr, y_ctr)
    temp[:, 4] = rois[:, 4] - 90
    rois = np.vstack((rois, temp))
    return rois


def aabb_2_obb(aabbs):
    """
    Append theta=0.0 at the end of aabbs
    """
    aabbs = np.array(aabbs)
    obbs = np.zeros((aabbs.shape[0], 5), dtype=np.float32)
    obbs[:, 0:4] = aabbs
    return obbs


def aabb_2_quad(aabbs):
    quads = np.zeros((aabbs.shape[0], 8), dtype=np.float32)
    quads[:, 0] = aabbs[:, 0]
    quads[:, 1] = aabbs[:, 1]
    quads[:, 2] = aabbs[:, 2]
    quads[:, 3] = aabbs[:, 1]
    quads[:, 4] = aabbs[:, 2]
    quads[:, 5] = aabbs[:, 3]
    quads[:, 6] = aabbs[:, 0]
    quads[:, 7] = aabbs[:, 3]
    return quads


def obb_2_quad(obb):
    obb = np.array(obb, dtype=np.float32)
    bbox = obb[0:4]
    ws = bbox[2] - bbox[0] + 1
    hs = bbox[3] - bbox[1] + 1
    x_ctr = bbox[0] + 0.5 * (ws - 1)
    y_ctr = bbox[1] + 0.5 * (hs - 1)
    # anticlockwise: +
    theta = math.radians(obb[4])
    bbox = bbox[np.newaxis, :]
    quad = aabb_2_quad(bbox).reshape(-1, 2)
    quad = quad - [x_ctr, y_ctr]
    affine_mat = [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    quad = np.dot(quad, affine_mat)
    quad = quad + [x_ctr, y_ctr]
    quad = quad.reshape(1, 8)
    return quad


def obb_2_aabb(obbs):
    quads = np.zeros((obbs.shape[0], 8), dtype=np.float32)
    for idx, obb in enumerate(obbs):
        quads[idx, :] = obb_2_quad(obb)
    aabbs = quad_2_aabb(quads)
    return aabbs


def quad_2_aabb(quads):
    aabbs = np.zeros((quads.shape[0], 4), dtype=np.float32)
    aabbs[:, 0] = np.min(quads[:, 0::2], 1)
    aabbs[:, 1] = np.min(quads[:, 1::2], 1)
    aabbs[:, 2] = np.max(quads[:, 0::2], 1)
    aabbs[:, 3] = np.max(quads[:, 1::2], 1)
    return aabbs


if __name__ == '__main__':
    temps = obb_2_quad(np.array([1, 1, 4, 2, 90], dtype=np.float32))
    print temps

