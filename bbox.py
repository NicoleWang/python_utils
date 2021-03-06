import sys, os, string
import cPickle
import numpy as np

def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float)
    #cdef DTYPE_t iw, ih, box_area
    #cdef DTYPE_t ua
    #cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def trans_txt_to_pkl(indir, namelist, outpath):
    with open(namelist, 'r') as f:
        names = [x.strip() for x in f.readlines()]
    all_boxes = []
    for name in names:
        txtpath = os.path.join(indir, name + ".txt")
        with open(txtpath, 'r') as f:
            lines = [[string.atoi(y) for y in x.strip().split()]
                    for x in f.readlines()]
        boxes = np.array(lines)
        all_boxes.append(boxes)
    with open(outpath, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

