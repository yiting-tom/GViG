# %%
from pandarallel import pandarallel

pandarallel.initialize()
import pandas as pd

from configs import consts as C
from configs import paths as P
from utils import file_handler as FH

# %%
filename = "test_public"
pri = pd.read_pickle(
    f"/home/P76104419/ICCV/dataset/base64/vqa/{filename}-with_ofa+mplug_ans.pickle"
)
# %%
pri["P2"] = pri["question"] + " " + pri["ofa"]
pri["P3"] = pri["question"] + " " + pri["mplug"]
pri["P4"] = pri["question"] + " " + pri["ofa"] + " . " + pri["mplug"]
pri["P5"] = (
    "question: " + pri["question"] + " hint: " + pri["ofa"] + " . " + pri["mplug"]
)
# %%
ofa_col = ["question_id", "image_id", "P2", "answer", "candidate", "image"]
mplug_col = ["question_id", "image_id", "P3", "answer", "candidate", "image"]
P4_col = ["question_id", "image_id", "P4", "answer", "candidate", "image"]
P5_col = ["question_id", "image_id", "P5", "answer", "candidate", "image"]
# %%
ofa = pri[ofa_col].rename(columns={"P2": "question"})
mplug = pri[mplug_col].rename(columns={"P3": "question"})
p4 = pri[P4_col].rename(columns={"P4": "question"})
p5 = pri[P5_col].rename(columns={"P5": "question"})
# %%
ofa_out = FH.df_vqa_to_vg(ofa, filename)
mplug_out = FH.df_vqa_to_vg(mplug, filename)
p4_out = FH.df_vqa_to_vg(p4, filename)
p5_out = FH.df_vqa_to_vg(p5, filename)
# %%
FH.to_vg_csv(ofa_out, f"{filename}-P2")
FH.to_vg_csv(mplug_out, f"{filename}-P3")
FH.to_vg_csv(p4_out, f"{filename}-P4")
FH.to_vg_csv(p5_out, f"{filename}-P5")
# %%
