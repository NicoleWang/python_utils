import sys
import os
import numpy as np
import string
import cPickle
from bbox import bbox_overlaps



if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: evaluate_text.py gt.pkl res.pkl outfile"
        exit()

    gtpath  = sys.argv[1]
    respath = sys.argv[2]
    outfile = sys.argv[3]
    area_thresh = 0.1
    #print os.path.splitext(gtpath)[1]
    #print os.path.splitext(respath)[1]
    if os.path.splitext(gtpath)[1] != '.pkl':
        print "gt must be in cPickle format\n call trans_txt_to_pkl first"
        exit()
    if os.path.splitext(respath)[1] != '.pkl':
        print "results must be in cPickle format, please check lib/fast_rcnn/test.py"
        exit()

    with open(gtpath, 'rb') as f:
        gt_boxes = cPickle.load(f)
    #gt_boxes[:, 2:3] = gt_boxes[:, 0:1] + gt_boxes[:, 2:3] - 1 ##### WRONG #######
    with open(respath, 'rb') as f:
        res_boxes = cPickle.load(f)

    gt_char_num = 0;
    res_char_num = 0;
    valid_gt_char_num = 0;
    valid_res_char_num = 0;
    #print gt_boxes[1]
    for i in xrange(0, len(gt_boxes)):
        print "processing %dth test sample" % i
        gt = gt_boxes[i]
        gt[:, 2] = gt[:, 0] + gt[:, 2] - 1
        gt[:, 3] = gt[:, 1] + gt[:, 3] - 1
        res = res_boxes[i]
        gt_char_num += gt.shape[0]
        res_char_num += res.shape[0]
        overlaps = bbox_overlaps(np.ascontiguousarray(res, dtype=np.float),
                             np.ascontiguousarray(gt, dtype=np.float))

        gt_argmax_overlap = overlaps.argmax(axis=0)
        gt_overlap = overlaps[gt_argmax_overlap, np.arange(gt.shape[0])]
        valid_gt = np.where(gt_overlap >= area_thresh)
        valid_gt_char_num += valid_gt[0].shape[0]

        res_argmax_overlap = overlaps.argmax(axis=1)
        res_overlap = overlaps[np.arange(res.shape[0]), res_argmax_overlap]
        valid_res = np.where(res_overlap >= area_thresh)
        valid_res_char_num += valid_res[0].shape[0]

    recall = valid_gt_char_num * 1.0 / gt_char_num
    precision = valid_res_char_num * 1.0 / res_char_num
    print "recall: %f \n precision: %f" %(recall, precision)
