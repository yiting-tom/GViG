# %%
import logging
import re
import unicodedata

import pandas as pd
from generate_wsdm_vg_data import RESULT_COLUMNS
from wsdm_data.utils import img_formatter

from configs import paths

# %%
test = pd.read_csv(
    "/home/P76104419/wsdm2023/VQA/dataset/test_public-annot.csv",
)


# %%
def format_to_vg_result_columns(df):
    image_path = "/home/P76104419/dataset/wsdm/test_public"
    df["unique_id"] = df.index
    df["image_id"] = df["image"].map(lambda x: x.split("/")[-1].split(".")[0])
    df["bbox"] = (
        df[["left", "top", "right", "bottom"]].astype(str).agg(",".join, axis=1)
    )
    df["image"] = df["image"].map(
        lambda x: img_formatter.url_to_base64(
            url=x,
            image_dir=image_path,
        )
    )
    df.drop(columns=["left", "top", "right", "bottom"], inplace=True)
    df.rename(columns={"question": "text"}, inplace=True)
    df.drop(columns=["width", "height"], inplace=True)
    return df[RESULT_COLUMNS]


# %%
df = format_to_vg_result_columns(df=test)
# %%
df
# %%
df.to_csv(
    "dataset/iccv/only_q_test_public-1705.tsv",
    index=False,
    sep="\t",
    header=False,
)
