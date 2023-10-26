# %%
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import file_handler as FH


def get_iou(bb1, bb2):
    # Taken from https://stackoverflow.com/a/42874377
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {0, 2, 1, 3}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {0, 2, 1, 3}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = np.array(bb1).astype(float)
    bb2 = np.array(bb2).astype(float)
    if bb1[0] > bb1[2]:
        bb1[0], bb1[2] = bb1[2], bb1[0]
    if bb1[1] > bb1[3]:
        bb1[1], bb1[3] = bb1[3], bb1[1]
    if bb1[0] == bb1[2] or bb1[1] == bb1[3]:
        return 0
    assert bb1[0] <= bb1[2], f"bb1[0] = {bb1[0]}, bb1[2] = {bb1[2]}"
    assert bb1[1] <= bb1[3], f"bb1[1] = {bb1[1]}, bb1[3] = {bb1[3]}"
    assert bb2[0] <= bb2[2]
    assert bb2[1] <= bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ghround-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# %%
split: Literal["public", "private"] = "private"
P = 4
ans = FH.read_vg_csv(f"test_{split}-P{P}.csv")["bbox"].map(eval)


# %%
def get_center(box):
    return (box[..., 0] + box[..., 2]) / 2, (box[..., 1] + box[..., 3]) / 2


def filter_outlier(coor, dev_threshold=0.4):
    mean = coor.mean(axis=1)
    std = coor.std(axis=1)
    z = abs(coor - mean.repeat(len(coor[0])).reshape(-1, len(coor[0])))
    mask = z < dev_threshold
    return mask


def ensemble(
    tag,
    ckpt_pair,
    weights,
    dev_threshold=2.5,
    method: Literal["union", "mean"] = "mean",
):
    result = []
    iou = [
        np.array(
            pd.read_json(
                f"/home/P76104419/ICCV/results/vg/vqa-P{tag}_dif-0/{ckpt}/{json}_predict.json"
            )["box"].values.tolist()
        )
        # ensemble ckpt tags
        for tag, ckpt, json in ckpt_pair
    ]

    # (Model, Data, Coordination)
    predicts = np.stack(iou)

    cen_x, cen_y = get_center(predicts)
    cen_x, cen_y = cen_x.T, cen_y.T

    # (Model, Data)
    (cen_x_mask, cen_y_mask) = (
        filter_outlier(cen_x, dev_threshold).T,
        filter_outlier(cen_y, dev_threshold).T,
    )

    mask = np.stack(
        [
            cen_x_mask,
            cen_y_mask,
            cen_x_mask,
            cen_y_mask,
        ],
        axis=-1,
    )

    result = predicts.copy()
    result[~mask] = np.nan
    if method == "mean":
        pred = np.average(
            result,
            axis=0,
            weights=weights,
        )
    elif method == "union":
        pred = np.stack(
            [
                result[..., 0].min(axis=0),
                result[..., 1].max(axis=0),
                result[..., 2].min(axis=0),
                result[..., 3].max(axis=0),
            ],
            axis=-1,
        )
    nan_mask = np.isnan(pred)
    pred[nan_mask] = predicts[0][nan_mask]
    result = []
    for p, a in zip(pred, ans):
        result.append(get_iou(p, a))
    return {tag: sum(result) / len(result)}


from copy import copy

from tqdm.notebook import tqdm_notebook

e1 = {}
input_args = []
ckpt_pair = [
    # (2,4),(1,1),(3,4),(4,4),(5,4),(5,5),(2,2),(3,4),(2,4) # exp1
    "2, ckpt_10_11500, private-P4",  # 1
    "1, ckpt_5_6000, private-P1",  # 2
    "3, ckpt_10_12000, private-P4",  # 3
    "4, ckpt_10_11000, private-P4",  # 4
    "5, ckpt_10_11000, private-P4",  # 5
    "5, ckpt_6_6500, private-P5",  # 6
    "4, ckpt_10_12000, private-P4",  # 7
    "5, ckpt_5_5000, private-P5",  # 8
    "5.w1.sh, ckpt_best, test_private-P5",  # 9
    "4.w1.sh, ckpt_best, test_private-P4",  # 10
]
ckpt_pair = [i.split(", ") for i in ckpt_pair]

# for idx in range(len(ckpt_list)):
for th in np.arange(5.2, 5.21, 0.02):
    for w1 in np.arange(0.4, 0.41, 0.01):
        for w2 in np.arange(0.15, 0.16, 0.02):
            for w3 in np.arange(0.1, 0.11, 0.02):
                for w4 in np.arange(0.30, 0.31, 0.02):
                    for w5 in np.arange(0.1, 0.11, 0.02):
                        for w6 in np.arange(0.70, 0.71, 0.02):
                            for w7 in np.arange(0.15, 0.16, 0.02):
                                for w8 in np.arange(0.3, 0.31, 0.02):
                                    for w9 in np.arange(0.6, 0.61, 0.02):
                                        for w10 in np.arange(0.5, 0.51, 0.02):
                                            weights = [
                                                w1,
                                                w2,
                                                w3,
                                                w4,
                                                w5,
                                                w6,
                                                w7,
                                                w8,
                                                w9,
                                                w10,
                                            ]
                                            tag = ", ".join(f"{w:.2f}" for w in weights)
                                            input_args.append(
                                                (
                                                    f"{tag}-{th:0.2f}",
                                                    copy(ckpt_pair),
                                                    copy(weights),
                                                    copy(th),
                                                )
                                            )

print(len(input_args))

from multiprocessing import Pool

out = Pool().starmap(ensemble, input_args)

out = {str(list(o.keys())[0]): list(o.values())[0] for o in out}
s1 = pd.Series(out)
s1.plot()
print(s1.sort_values(ascending=True)[-5:])
print("  w1    w2    w3    w4    w5    w6    w7    w8")
# %%
ens_result = {}
for th in np.arange(0.5, 3.0, 0.1):
    ens_result[f"{th:0.2f}"] = ensemble(ckpt_list, th)

# %%
import matplotlib.pyplot as plt

pd.Series(ens_result).max()
# %%
res = np.array(result)
correct_mask = res > 0.5


def get_area(box):
    box = np.array(box)
    return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])


incorrect_ans_area = ans[~correct_mask].map(get_area)
incorrect_pred_area = pd.DataFrame(get_area(pred[~correct_mask]))
# %%
incorrect_ans_area.plot()
incorrect_pred_area.plot()
# %%
print("ans_area:", incorrect_ans_area.describe(), sep="\n")
print("pred_area:", incorrect_pred_area.describe(), sep="\n")
print(iou)
# %%
len(pred[~correct_mask])
