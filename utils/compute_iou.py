# %%
from typing import Literal

import numpy as np
import pandas as pd

from configs import paths as P
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
    assert bb1[0] <= bb1[2]
    assert bb1[1] <= bb1[3]
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
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# %%
iou = {}
data = []
split: Literal["test_public", "test_private"] = "test_private"
exp_tag = "basic_prompt"
model_sizes = ["tiny"]
trainP_list = [4]
valP_list = [4]
df_columns = ["trainP", "valP", "iou"]
vg_file_template = "{exp_tag}/{split}-P{valP}.csv"
result_file_template = str(
    P.ROOT / "results/vg/{exp_tag}/P{trainP}/{split}-P{valP}_predict.json"
)
# %%
for model_size in model_sizes:
    for valP in valP_list:
        # read ground truth
        ans = FH.read_vg_csv(
            vg_file_template.format(
                exp_tag=exp_tag, model_size=model_size, split=split, valP=valP
            )
        )["bbox"].map(eval)
        for trainP in trainP_list:
            # read prediction
            pre = pd.read_json(
                result_file_template.format(
                    exp_tag=exp_tag,
                    model_size=model_size,
                    split=split,
                    valP=valP,
                    trainP=trainP,
                )
            )["box"]
            # compute iou
            result = [get_iou(p, a) for p, a in zip(pre, ans)]
            data.append(
                [
                    trainP,
                    valP,
                    # model_size,
                    sum(result) / len(result),
                ]
            )
# %%
import matplotlib.pyplot as plt

# %%
df = pd.DataFrame(data, columns=df_columns)
df
# %%
df.sort_values(by=["iou"])
# %%
df.sort_index(inplace=True)
# %%
df
# %%
df.to_csv("basic_prompt.csv", index=False)
