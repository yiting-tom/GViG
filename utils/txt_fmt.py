import logging
import re
import unicodedata

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

import pandas as pd

from configs import consts, paths

L: logging.Logger = logging.getLogger(logging.basicConfig(level=logging.INFO))


def df_ltrb_to_bbox(df: pd.DataFrame) -> pd.DataFrame:
    df["bbox"] = (
        df[["left", "top", "right", "bottom"]].astype(str).agg(",".join, axis=1)
    )

    df.drop(
        columns=["left", "top", "right", "bottom", "height", "width"],
        inplace=True,
        errors="ignore",
    )
    return df


def full_width_to_half_width(s: str) -> str:
    """full-width to half-width and translate to ascii with unicodedata"""
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )


def translate_head_punctuations(s: str) -> str:
    """translate head punctuations to specific format"""
    sp = s.split(" ")
    x = sp[0]
    x = re.sub(r"w[wh]h", "wh", x)
    x = re.sub(r"\'s", " is", x)
    x = re.sub(r"\'re", " are", x)
    x = re.sub(r"\'|\"|-|_|\,|/|\.", "", x)
    sp[0] = x
    result = " ".join(sp).lower()
    return result.lower()


def reformat_question(q: str) -> str:
    """reformat question to specific format"""
    q = q.lower()
    q = full_width_to_half_width(q)
    q = translate_head_punctuations(q)
    if not q.endswith("?"):
        q += "?"
    while q.endswith("??"):
        q = q[:-1]
    if len(q.split("?")) > 2:
        q = q.split("?")[0] + "?"
    q = q.capitalize()

    return q


def df_format_question(df: pd.DataFrame) -> pd.DataFrame:
    df["question"] = df["question"].parallel_apply(reformat_question)
    return df


def df_columns_vg_format(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        df["text"] = df["question"]
        df.drop("question", errors="ignore", inplace=True)
    if "unique_id" not in df.columns:
        df["unique_id"] = df.index
    if "image_id" not in df.columns:
        df["image_id"] = df.index
    return df[consts.VG_COLUMNS]
