# %%
# %% generate prob1 dataset
import random

import numpy as np
import pandas as pd

from configs import consts as C
from configs import paths as P
from utils import file_handler as FH

random.seed(42)
np.random.seed(42)
# %%
for P in range(1, 6):
    pri = FH.read_vg_csv(f"test_private-P{P}")
    pub = FH.read_vg_csv(f"test_public-P{P}")
    test = pd.concat([pri, pub])
    pri_new = test.sample(n=len(pri))
    pub_new = test.drop(pri_new.index)
    FH.to_vg_csv(pri_new, f"prob1-test_private-P{P}")
    FH.to_vg_csv(pub_new, f"prob1-test_public-P{P}")
# %%
split = "private"
df = pd.read_pickle(P.VG_64 / f"prob1_test_{split}.pkl")
mplug_voc = np.load(P.DATASET / "vocab" / "mplug_vocab.npy", allow_pickle=True)
ofa_voc = np.load(P.DATASET / "vocab" / "ofa_vocab.npy", allow_pickle=True)
ofa_random = np.random.choice(ofa_voc, len(df))
mplug_random = np.random.choice(mplug_voc, len(df))
# %%
df
# %%
mplug = pd.read_json(f"/home/P76104419/ICCV/dataset/gqa/mplug_predict-{split}.json")
ofa = pd.read_json(f"/home/P76104419/ICCV/dataset/gqa/ofa_predict-{split}.json")
df["mplug"] = mplug["text"]
df["ofa"] = ofa["answer"]
# %%
df["text"] = df["text"].apply(lambda x: x.replace("?", " ? "))
df["P1"] = df["text"]
df["P2"] = df["text"] + df["ofa"]
# %%
df["P3"] = df["text"] + df["mplug"]
df["P4"] = df["text"] + df["ofa"] + " . " + df["mplug"]
df["P5"] = "question: " + df["text"] + " hint: " + df["mplug"] + " . " + df["ofa"]
# %%
for p in [1, 2, 3, 4, 5]:
    df[["unique_id", "image_id", f"P{p}", "bbox", "image"]].to_csv(
        P.VG_64 / f"gqa_{split}-P{p}.csv",
        sep="\t",
        index=False,
        header=False,
    )
# %%
df["P4.1"] = df["P4"] + " . " + ofa_random
df["P4.2"] = df["P4"] + " . " + ofa_random + " . " + mplug_random
df["P5.0"] = "question: " + df["text"] + " hint: "
df["P5.1"] = "question: " + df["text"] + " hint: " + ofa_random
df["P5.2"] = "question: " + df["text"] + " hint: " + ofa_random + " . " + mplug_random
# %%
for p in ["4.1", "4.2", "5.0", "5.1", "5.2"]:
    df[["unique_id", "image_id", f"P{p}", "bbox", "image"]].to_csv(
        P.VG_64 / f"prob1_test_{split}-P{p}.csv",
        sep="\t",
        index=False,
        header=False,
    )
# %%
df[["unique_id", "image_id", "text", "bbox", "image", "ofa", "mplug"]].to_pickle(
    P.VG_64 / f"gqa_{split}-entire.pkl"
)
