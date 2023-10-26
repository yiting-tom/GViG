import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from configs import consts as C
from configs import paths as P

L = logging.getLogger(logging.basicConfig(level=logging.INFO))


def read_vqa_csv(split: str):
    filepath = P.BASE64_DIR / "vqa" / f"{split}.csv"
    try:
        L.info(f"Loading {filepath}")
        return pd.read_csv(filepath, sep="\t", names=C.VQA_COLUMNS)
    except FileNotFoundError as e:
        L.info(e)


def read_vg_csv(split: str, dir: Optional[str] = None):
    filepath = (
        P.BASE64_DIR / "vg" / f"{split}" if dir is None else Path(dir) / f"{split}"
    )

    if filepath.suffix == "":
        filepath = filepath.with_suffix(".csv")
    try:
        L.info(f"Loading {filepath}")
        return pd.read_csv(filepath, sep="\t", names=C.VG_COLUMNS)
    except FileNotFoundError as e:
        L.info(e)


def to_vqa_csv(df: pd.DataFrame, split: str):
    df[C.VQA_COLUMNS].to_csv(
        P.BASE64_DIR / "vqa" / f"{split}.csv",
        index=False,
        sep="\t",
        header=False,
    )


def to_vg_csv(df: pd.DataFrame, split: str):
    if not split.endswith(".csv"):
        split = f"{split}.csv"
    df[C.VG_COLUMNS].to_csv(
        P.BASE64_DIR / "vg" / split,
        index=False,
        sep="\t",
        header=False,
    )


def df_vqa_to_vg(df: pd.DataFrame, split: str):
    assert set(df.columns) == set(C.VQA_COLUMNS)
    from utils.txt_fmt import df_ltrb_to_bbox

    if split in ("train", "test_public", "test_private"):
        official_vg = pd.read_csv(P.OFFICIAL / f"{split}.csv")
    elif split in ("train-fus"):
        official_vg = pd.read_csv(P.OFFICIAL / f"train.csv")
        official_vg = pd.concat([official_vg, official_vg], axis=0)

    official_vg = df_ltrb_to_bbox(official_vg)
    df["unique_id"] = df["question_id"]
    df["text"] = df["question"]
    df["bbox"] = official_vg["bbox"]
    return df[C.VG_COLUMNS]


def df_vg_to_vqa(df: pd.DataFrame):
    assert set(df.columns) == set(C.VG_COLUMNS)
    df["question_id"] = df["unique_id"]
    df["question"] = df["text"]
    df["answer"] = "0.0|!+0"
    df["candidate"] = ""
    return df[C.VQA_COLUMNS]
