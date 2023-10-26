# %%
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=32)
from pathlib import Path

import pandas as pd

from configs import consts as C
from configs import paths as P
from utils import file_handler as FH
from utils import img_fmt as IF
from utils import txt_fmt as TF

# %%
img_path = P.GQA_DIR / "images"
split = "train"
df = pd.read_json(P.GQA_DIR / "annotations" / f"{split}.json")
df["image"] = pd.read_pickle(P.GQA_DIR / f"{split}-base64.pkl")["image"]
# %%
df["image"] = df["filename"].parallel_map(lambda x: IF.filepath_to_base64(img_path / x))
# %%
df["bbox"] = df["bbox"].map(lambda x: ",".join(map(str, x)))
df.drop(columns=["width", "height", "filename"], inplace=True, errors="ignore")
df.rename(columns={"refer": "question"}, inplace=True)
df = TF.df_format_question(df)
df = TF.df_columns_vg_format(df)
# %%
df.to_pickle(P.VG_64 / f"gqa_{split}.pkl")
# %%
FH.to_vg_csv(df, f"gqa_{split}-P1")
# %%
df = FH.df_vg_to_vqa(df)
# %%
df[C.VQA_COLUMNS].to_csv(P.VQA_64 / "gqa_train.csv")
# %%
FH.to_vqa_csv(df[C.VQA_COLUMNS], P.VQA_64 / "gqa_val.csv")
# %%
ofa = pd.read_json(
    "/home/P76104419/ICCV/results/vqa_zeroshot-GQA_train-cand/_predict.json"
)["answer"]
# %%
df["ofa"] = ofa
# %%
df.to_pickle(P.VG_64 / f"gqa_{split}-ofa.pkl")
