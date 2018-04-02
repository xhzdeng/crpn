# --------------------------------------------------------
# CRPN
# Written by Linjie Deng
# --------------------------------------------------------
import yaml
import caffe
import numpy as np
from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms
from quad.quad_convert import whctrs, mkanchors, quad_2_aabb, obb_2_quad, dual_roi
from quad.quad_2_obb import quad_2_obb

DEBUG = False


class Corner(object):
    # Corner property
    def __init__(self, name):
        self.name = name
        # position
        self.pos = None
        # probability
        self.prb = None
        # class of link direction
        self.cls = None


class ProposalLayer(caffe.Layer):
    # Corner-based Region Proposal Network
    # Input: prob map of each corner
    # Output: quadrilateral region proposals
    def setup(self, bottom, top):
        # top: (ind, x1, y1, x2, y2, x3, y3, x4, y4)
        layer_params = yaml.load(self.param_str)
        self._feat_stride = layer_params['feat_stride']
        num_rois = 2 if cfg.DUAL_ROI else 1
        top[0].reshape(num_rois, 9)
        if len(top) > 1:
            top[1].reshape(num_rois, 5)

    def forward(self, bottom, top):
        # params
        cfg_key = self.phase  # either 'TRAIN' or 'TEST'
        if cfg_key == 0:
            cfg_ = cfg.TRAIN
        else:
            cfg_ = cfg.TEST

        # corner params
        pt_thres = cfg_.PT_THRESH
        pt_max_num = cfg.PT_MAX_NUM
        pt_nms_range = cfg.PT_NMS_RANGE
        pt_nms_thres = cfg.PT_NMS_THRESH
        # proposal params
        ld_interval = cfg.LD_INTERVAL
        ld_um_thres = cfg.LD_UM_THRESH
        # rpn params
        # min_size = cfg_.RPN_MIN_SIZE
        nms_thresh = cfg_.RPN_NMS_THRESH
        pre_nms_topN = cfg_.RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg_.RPN_POST_NMS_TOP_N

        im_info = bottom[0].data[0, :]
        score_tl = bottom[1].data[0, :].transpose((1, 2, 0))
        score_tr = bottom[2].data[0, :].transpose((1, 2, 0))
        score_br = bottom[3].data[0, :].transpose((1, 2, 0))
        score_bl = bottom[4].data[0, :].transpose((1, 2, 0))
        scores = np.concatenate([score_tl[:, :, :, np.newaxis],
                                 score_tr[:, :, :, np.newaxis],
                                 score_br[:, :, :, np.newaxis],
                                 score_bl[:, :, :, np.newaxis]], axis=3)

        map_info = scores.shape[:2]
        # 1. sample corner candidates from prob maps
        tl, tr, br, bl = _corner_sampling(scores, pt_thres, pt_max_num, pt_nms_range, pt_nms_thres)
        # 2. assemble corner candidates into proposals
        proposals = _proposal_sampling(tl, tr, br, bl, map_info, ld_interval, ld_um_thres)
        # 3. filter
        proposals = filter_quads(proposals)
        scores = proposals[:, 8]
        proposals = proposals[:, :8]
        # 3. rescale quads into raw image space
        proposals = proposals * self._feat_stride
        # 4. quadrilateral non-max surpression
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]
        keep = nms(np.hstack((proposals, scores[:, np.newaxis])).astype(np.float32, copy=False), nms_thresh)
        proposals = proposals[keep, :]
        scores = scores[keep]
        if post_nms_topN > 0:
            proposals = proposals[:post_nms_topN, :]
            scores = scores[:post_nms_topN]
        if proposals.shape[0] == 0:
            # add whole image to avoid error
            print 'NO PROPOSALS!'
            proposals = np.array([[0, 0, im_info[1], 0, im_info[1], im_info[0], 0, im_info[0]]])
            scores = np.array([0.0])

        # output
        # top[0]: quads(x1, y1, x2, y2, x3, y3, x4, y4)
        # top[1]: rois(xmin, ymin, xmax, ymax, theta)
        # top[2]: scores
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*blob.shape)
        top[0].data[...] = blob
        if len(top) > 1:
            if cfg.DUAL_ROI:
                rois = quad_2_obb(np.array(proposals, dtype=np.float32))
                rois = dual_roi(rois)
            else:
                rois = quad_2_obb(np.array(proposals, dtype=np.float32))
            batch_inds = np.zeros((rois.shape[0], 1), dtype=np.float32)
            blob = np.hstack((batch_inds, rois.astype(np.float32, copy=False)))
            top[1].reshape(*blob.shape)
            top[1].data[...] = blob
        if len(top) > 2:
            scores = np.vstack((scores, scores)).transpose()
            top[2].reshape(*scores.shape)
            top[2].data[...] = scores

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _map_2_corner(pred_map, thresh, max_num, nms_range, nms_thres):
    pos_map = 1 - pred_map[:, :, 0]
    pts_cls = np.argmax(pred_map[:, :, 1:], 2) + 1
    ctr_y, ctr_x = np.where(pos_map >= thresh)
    ctr_pts = np.vstack((ctr_x, ctr_y)).transpose()
    ws = np.ones(ctr_x.shape) * nms_range
    hs = np.ones(ctr_y.shape) * nms_range
    anchors = np.hstack((mkanchors(ws, hs, ctr_x, ctr_y), get_value(ctr_pts, pos_map)))
    keep = nms(anchors, nms_thres)
    if max_num > 0:
        keep = keep[:max_num]
    pos = ctr_pts[keep, :]
    prb = pos_map
    cls = pts_cls
    return pos, prb, cls


def _corner_sampling(maps, thresh, max_num, nms_range, nms_thres):
    tl = Corner('top_left')
    tl.pos, tl.prb, tl.cls = _map_2_corner(maps[:, :, :, 0], thresh, max_num, nms_range, nms_thres)
    tr = Corner('top_right')
    tr.pos, tr.prb, tr.cls = _map_2_corner(maps[:, :, :, 1], thresh, max_num, nms_range, nms_thres)
    br = Corner('bot_right')
    br.pos, br.prb, br.cls = _map_2_corner(maps[:, :, :, 2], thresh, max_num, nms_range, nms_thres)
    bl = Corner('bot_left')
    bl.pos, bl.prb, bl.cls = _map_2_corner(maps[:, :, :, 3], thresh, max_num, nms_range, nms_thres)
    return tl, tr, br, bl


def _gen_diags(a, b, theta_invl=15, max_diff=1):
    max_label = round(360.0 / theta_invl)
    idx_a = np.arange(0, a.pos.shape[0])
    idx_b = np.arange(0, b.pos.shape[0])
    idx_a, idx_b = np.meshgrid(idx_a, idx_b)
    idx_a = idx_a.ravel()
    idx_b = idx_b.ravel()
    diag_pos = np.hstack((a.pos[idx_a, :], b.pos[idx_b, :]))
    #
    keep = np.where((diag_pos[:, 0] != diag_pos[:, 2]) | (diag_pos[:, 1] != diag_pos[:, 3]))[0]
    diag_pos = diag_pos[keep, :]
    prac_label = compute_link(diag_pos[:, 0:2], diag_pos[:, 2:4], theta_invl)
    pred_label = get_value(diag_pos[:, 0:2], a.cls)
    diff_label_a = diff_link(prac_label, pred_label, max_label)
    #
    prac_label = np.mod(prac_label + max_label / 2, max_label)
    pred_label = get_value(diag_pos[:, 2:4], b.cls)
    diff_label_b = diff_link(prac_label, pred_label, max_label)
    keep = np.where((diff_label_a <= max_diff) & (diff_label_b <= max_diff))[0]
    diag_pos = diag_pos[keep, :]
    diag_prb = np.hstack((get_value(diag_pos[:, 0:2], a.prb), get_value(diag_pos[:, 2:4], b.prb)))
    return diag_pos, diag_prb


def _gen_trias(diag_pos, diag_prb, c, theta_invl=15, max_diff=1):
    max_label = 360 / theta_invl
    idx_a = np.arange(0, diag_pos.shape[0])
    idx_b = np.arange(0, c.pos.shape[0])
    idx_a, idx_b = np.meshgrid(idx_a, idx_b)
    idx_a = idx_a.ravel()
    idx_b = idx_b.ravel()
    tria_pos = np.hstack((diag_pos[idx_a, :], c.pos[idx_b, :]))
    tria_prb = np.hstack((diag_prb[idx_a, :], get_value(c.pos[idx_b, :], c.prb)))
    #
    areas = compute_tria_area(tria_pos[:, 0:2], tria_pos[:, 2:4], tria_pos[:, 4:6])
    keep = np.where(areas != 0)[0]
    tria_pos = tria_pos[keep, :]
    tria_prb = tria_prb[keep, :]
    ws, hs, ctr_x, ctr_y = whctrs(tria_pos[:, 0:4])
    prac_theta = compute_theta(tria_pos[:, 4:6], np.vstack((ctr_x, ctr_y)).transpose())
    prac_label = np.floor(prac_theta / theta_invl) + 1
    pred_label = get_value(tria_pos[:, 4:6], c.cls)
    diff_label = diff_link(prac_label, pred_label, max_label)
    keep = np.where(diff_label <= max_diff)[0]
    tria_pos = tria_pos[keep, :]
    tria_prb = tria_prb[keep, :]
    prac_theta = prac_theta[keep]
    #
    prac_theta = np.mod(prac_theta + 180.0, 360.0) / 180.0 * np.pi
    len_diag = np.sqrt(np.sum(np.square(tria_pos[:, 0:2] - tria_pos[:, 2:4]), axis=1)) / 2.
    dist_x = len_diag * np.cos(prac_theta[:, 0])
    dist_y = len_diag * np.sin(prac_theta[:, 0])
    ws, hs, ctr_x, ctr_y = whctrs(tria_pos[:, 0:4])
    tria_pos[:, 4:6] = np.vstack((ctr_x + dist_x, ctr_y - dist_y)).astype(np.int32, copy=False).transpose()
    return tria_pos, tria_prb


def _get_last_one(tria, d):
    map_shape = d.prb.shape[:2]
    ws, hs, ctr_x, ctr_y = whctrs(tria[:, 0:4])
    pos = np.vstack((2 * ctr_x - tria[:, 4], 2 * ctr_y - tria[:, 5])).transpose()
    pos[:, 0] = np.maximum(np.minimum(pos[:, 0], map_shape[1] - 1), 0)
    pos[:, 1] = np.maximum(np.minimum(pos[:, 1], map_shape[0] - 1), 0)
    pos = np.array(pos, dtype=np.int32)
    prb = get_value(pos, d.prb)
    return pos, prb


def _clip_trias(tria_pos, tria_prb, c, map_info):
    tria_pos[:, 4] = np.maximum(np.minimum(tria_pos[:, 4], map_info[1] - 1), 0)
    tria_pos[:, 5] = np.maximum(np.minimum(tria_pos[:, 5], map_info[0] - 1), 0)
    tria_prb[:, 2:] = get_value(tria_pos[:, 4:6], c.prb)
    return tria_pos, tria_prb


def _proposal_sampling(tl, tr, br, bl, map_info, theta_invl=15, max_diff=1):
    # DIAG: [top_left, bot_right]
    diag_pos, diag_prb = _gen_diags(tl, br, theta_invl, max_diff)
    # TRIA: [DIAG, top_right]
    tria_pos, tria_prb = _gen_trias(diag_pos, diag_prb, tr, theta_invl, max_diff)
    # QUAD: [TRIA, bot_left]
    temp_pos, temp_prb = _get_last_one(tria_pos, bl)
    # refine top_right
    tria_pos, tria_prb = _clip_trias(tria_pos, tria_prb, tr, map_info)
    # assemble
    score = compute_score(np.hstack((tria_prb, temp_prb)))
    quads = np.hstack((tria_pos[:, 0:2], tria_pos[:, 4:6], tria_pos[:, 2:4], temp_pos))
    quads = np.hstack((quads, score[:, np.newaxis]))

    # TRIA: [DIAG, bot_left]
    tria_pos, tria_prb = _gen_trias(diag_pos, diag_prb, bl, theta_invl, max_diff)
    # QUAD: [TRIA, top_right]
    temp_pos, temp_prb = _get_last_one(tria_pos, tr)
    # refine bot_left
    tria_pos, tria_prb = _clip_trias(tria_pos, tria_prb, bl, map_info)
    # assemble
    score = compute_score(np.hstack((tria_prb, temp_prb)))
    quad = np.hstack((tria_pos[:, 0:2], temp_pos, tria_pos[:, 2:4], tria_pos[:, 4:6]))
    quad = np.hstack((quad, score[:, np.newaxis]))
    quads = np.vstack((quads, quad))

    # DIAG: [bot_left, top_right]
    diag_pos, diag_prb = _gen_diags(bl, tr, theta_invl, max_diff)
    # TRIA: [DIAG, top_left]
    tria_pos, tria_prb = _gen_trias(diag_pos, diag_prb, tl, theta_invl, max_diff)
    # QUAD: [TRIA, bot_right]
    temp_pos, temp_prb = _get_last_one(tria_pos, br)
    # refine top_left
    tria_pos, tria_prb = _clip_trias(tria_pos, tria_prb, tl, map_info)
    # assemble
    score = compute_score(np.hstack((tria_prb, temp_prb)))
    quad = np.hstack((tria_pos[:, 4:6], tria_pos[:, 2:4], temp_pos, tria_pos[:, 0:2]))
    quad = np.hstack((quad, score[:, np.newaxis]))
    quads = np.vstack((quads, quad))

    # TRIA: [DIAG, bor_right]
    tria_pos, tria_prb = _gen_trias(diag_pos, diag_prb, br, theta_invl, max_diff)
    # QUAD: [TRIA, top_left]
    temp_pos, temp_prb = _get_last_one(tria_pos, tl)
    # refine bor_right
    tria_pos, tria_prb = _clip_trias(tria_pos, tria_prb, br, map_info)
    # assemble
    score = compute_score(np.hstack((tria_prb, temp_prb)))
    quad = np.hstack((tria_pos[:, 0:2], temp_pos, tria_pos[:, 2:4], tria_pos[:, 4:6]))
    quad = np.hstack((quad, score[:, np.newaxis]))
    quads = np.vstack((quads, quad))

    return quads


def get_value(pts, maps):
    vals = maps[pts[:, 1], pts[:, 0]]
    return vals[:, np.newaxis]


def compute_score(scores):
    score = scores[:, 0] * scores[:, 1] * scores[:, 2] * scores[:, 3]
    return score


def compute_theta(p1, p2):
    dx = p2[:, 0] - p1[:, 0]
    dy = p2[:, 1] - p1[:, 1]
    val = dx / np.sqrt(dx * dx + dy * dy)
    val = np.maximum(np.minimum(val, 1), -1)
    theta = np.arccos(val) / np.pi * 180
    idx = np.where(dy > 0)[0]
    theta[idx] = 360 - theta[idx]
    return theta[:, np.newaxis]


def compute_link(p1, p2, interval):
    theta = compute_theta(p1, p2)
    label = np.floor(theta / interval) + 1
    return label


def diff_link(t1, t2, max_orient):
    dt = np.abs(t2 - t1)
    dt = np.minimum(dt, max_orient - dt)
    return dt


def compute_tria_area(p1, p2, p3):
    area = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
           (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return area


def filter_quads(quads):
    area_1 = compute_tria_area(quads[:, 0:2], quads[:, 2:4], quads[:, 4:6])
    area_2 = compute_tria_area(quads[:, 0:2], quads[:, 2:4], quads[:, 6:8])
    area_3 = compute_tria_area(quads[:, 0:2], quads[:, 4:6], quads[:, 6:8])
    area_4 = compute_tria_area(quads[:, 2:4], quads[:, 4:6], quads[:, 6:8])
    areas = area_1 * area_2 * area_3 * area_4
    keep = np.where(areas != 0)[0]
    quads = quads[keep, :]
    return quads
