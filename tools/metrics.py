from typing import Dict, List, Optional, Tuple, Union
import multiprocessing
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch import BoolTensor, FloatTensor


Layout = Tuple[np.ndarray, np.ndarray]

# set True to disable parallel computing by multiprocessing (typically for debug)
# DISABLED = False
DISABLED = True


def compute_iou(
    box_1: Union[np.ndarray, FloatTensor],
    box_2: Union[np.ndarray, FloatTensor],
    generalized: bool = False,
) -> Union[np.ndarray, FloatTensor]:
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, FloatTensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    l1, t1, r1, b1 = box_1.T
    l2, t2, r2, b2 = box_2.T
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max), lib.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    if not generalized:
        return iou

    # outer region
    l_min = lib.minimum(l1, l2)
    r_max = lib.maximum(r1, r2)
    t_min = lib.minimum(t1, t2)
    b_max = lib.maximum(b1, b2)
    ac = (r_max - l_min) * (b_max - t_min)

    giou = iou - (ac - au) / ac

    return giou

def __compute_maximum_iou_for_layout(layouts_1_layouts_2: Tuple[List[Layout]], ) -> float:
    layout_1, layout_2 = layouts_1_layouts_2
    score = 0.0
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)
    for l in list(set(li.tolist())):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]
        n = len(_bi)
        m = len(_bj)
        if m > 0:
            ii, jj = np.meshgrid(range(n), range(m))
            ii, jj = ii.flatten(), jj.flatten()
            iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, m)
            # note: maximize is supported only when scipy >= 1.4
            ii, jj = linear_sum_assignment(iou, maximize=True)
            score += iou[ii, jj].sum().item()
    return score / N

def compute_maximum_iou(
    layouts_gt: List[Layout],
    layouts_generated: List[Layout],
    disable_parallel: bool = DISABLED,
    n_jobs: Optional[int] = None,
):
    args = list(zip(layouts_gt, layouts_generated))
    if disable_parallel:
        scores = []
        for arg in args:
            scores.append(__compute_maximum_iou_for_layout(arg))
    else:
        with multiprocessing.Pool(n_jobs) as p:
            scores = p.map(__compute_maximum_iou_for_layout, args)
    return np.array(scores)

def __compute_bbox_sim(
    bboxes_1: np.ndarray,
    category_1: np.int64,
    bboxes_2: np.ndarray,
    category_2: np.int64,
    C_S: float = 2.0,
    C: float = 0.5,
) -> float:
    # bboxes from diffrent categories never match
    if category_1 != category_2:
        return 0.0

    cx1, cy1, w1, h1 = bboxes_1
    cx2, cy2, w2, h2 = bboxes_2

    delta_c = np.sqrt(np.power(cx1 - cx2, 2) + np.power(cy1 - cy2, 2))
    delta_s = np.abs(w1 - w2) + np.abs(h1 - h2)
    area = np.minimum(w1 * h1, w2 * h2)
    alpha = np.power(np.clip(area, 0.0, None), C)

    weight = alpha * np.power(2.0, -1.0 * delta_c - C_S * delta_s)
    return weight

def __compute_docsim_between_two_layouts(
    layouts_1_layouts_2: Tuple[List[Layout]],
    max_diff_thresh: int = 3,
) -> float:
    layouts_1, layouts_2 = layouts_1_layouts_2
    bboxes_1, categories_1 = layouts_1
    bboxes_2, categories_2 = layouts_2

    N, M = len(bboxes_1), len(bboxes_2)
    if N >= M + max_diff_thresh or N <= M - max_diff_thresh:
        return 0.0

    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray(
        [
            __compute_bbox_sim(
                bboxes_1[i], categories_1[i], bboxes_2[j], categories_2[j]
            )
            for i, j in zip(ii, jj)
        ]
    ).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)

    if len(scores[ii, jj]) == 0:
        # sometimes, predicted bboxes are somehow filtered.
        return 0.0
    else:
        return scores[ii, jj].mean()

def compute_docsim(
    layouts_gt: List[Layout],
    layouts_generated: List[Layout],
    disable_parallel: bool = DISABLED,
    n_jobs: Optional[int] = None,
) -> float:
    """
    Compute layout-to-layout similarity and average over layout pairs.
    Note that this is different from layouts-to-layouts similarity.
    """
    args = list(zip(layouts_gt, layouts_generated))
    if disable_parallel:
        scores = []
        for arg in args:
            scores.append(__compute_docsim_between_two_layouts(arg))
    else:
        with multiprocessing.Pool(n_jobs) as p:
            scores = p.map(__compute_docsim_between_two_layouts, args)
    return np.array(scores)