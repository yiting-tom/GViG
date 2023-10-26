# %%

import logging
import re
import unicodedata

import pandas as pd
from pandarallel import pandarallel

from configs import paths

pandarallel.initialize(progress_bar=True)
import pandas as pd

from configs import consts as C
from configs import paths as P
from utils import img_fmt, txt_fmt

filename = "train"
test_pri = P.DATASET / "official" / f"{filename}.csv"
test_pic = P.DATASET / "official" / "images" / filename

df = pd.read_csv(test_pri)

# url -> base64
df["image"] = df["image"].parallel_map(lambda x: img_fmt.url_to_base64(x, test_pic))

# [left, top, right, bottom, height, width] -> [bbox]
df = txt_fmt.df_ltrb_to_bbox(df)

df = txt_fmt.df_format_question(df)

df = txt_fmt.df_columns_vg_format(df)

# %%
df[C.VG_COLUMNS].to_pickle(P.DATASET / "base64" / f"{filename}-vg.pkl")
df[C.VG_COLUMNS].to_csv(
    P.DATASET / "base64" / f"{filename}.csv",
    index=False,
    sep="\t",
    header=False,
)
# %%
train = pd.read_csv(
    "/home/P76104419/wsdm2023/VQA/dataset/2023/vg_input/final_train.csv",
    sep="\t",
    header=None,
    names=C.VG_COLUMNS,
)
# %%
tdf = train[len(df) :]
# %%
tdf["text"] = tdf["text"].parallel_map(lambda x: x.split("?")[0] + "?")
# %%
# %%
filename = "train"
df = pd.read_pickle(P.DATASET / "base64" / f"{filename}-vg.pkl")
# %%
df["unique_id"] = df.index
df["image_id"] = df.index
# %%
df["text"] = df["text"].parallel_map(lambda x: x.replace("?", " ?"))
# %%
df[C.VG_COLUMNS].to_pickle(P.DATASET / "base64" / f"{filename}-vg.pkl")
df[C.VG_COLUMNS].to_csv(
    P.DATASET / "base64" / f"{filename}.csv",
    index=False,
    sep="\t",
    header=False,
)
# %%
tdf
filename = "train-aug"
tdf.to_pickle(P.DATASET / "base64" / f"{filename}-vg.pkl")
tdf.to_csv(
    P.DATASET / "base64" / f"{filename}.csv",
    index=False,
    sep="\t",
    header=False,
)
