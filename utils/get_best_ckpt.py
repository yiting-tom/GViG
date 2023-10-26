# %%
from pathlib import Path

import numpy as np
import pandas as pd

from utils import file_handler as FH


# %%
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
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# %%
split = "private"
P = 4
ans = FH.read_vg_csv(f"test_{split}-P{P}.csv")["bbox"].map(eval)
result_folder = Path("/home/P76104419/ICCV/results/vg")
from collections import defaultdict

# %%
from tqdm import tqdm

result = defaultdict(list)
for tag in tqdm(range(1, 6)):
    folder = result_folder / f"vqa-P{tag}_dif-0"
    for sub_folder in folder.glob("ckpt*"):
        for P in range(1, 6):
            pre = pd.read_json(sub_folder / f"private-P{P}_predict.json")[
                "box"
            ].values.tolist()
            for p, a in zip(pre, ans):
                result[f"{tag}, {sub_folder.name}, P{P}"].append(get_iou(p, a))
# %%
output = {k: sum(v) / len(v) for k, v in result.items()}
# %%
from pprint import pprint

pprint(sorted(output.items(), key=lambda x: x[1], reverse=True))
