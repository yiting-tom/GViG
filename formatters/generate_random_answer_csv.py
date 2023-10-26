# %%
import numpy as np
import pandas as pd

from configs import consts as C
from configs import paths as P
from utils import file_handler as FH

np.random.seed(42)
# %%
split = "test_public"
p4 = FH.read_vg_csv(f"{split}-P4")
p5 = FH.read_vg_csv(f"{split}-P5")
ofa_vocab = np.load(P.DATASET / "vocab" / "ofa_vocab.npy", allow_pickle=True)
mplug_vocab = np.load(P.DATASET / "vocab" / "mplug_vocab.npy", allow_pickle=True)
ofa_random = np.random.choice(ofa_vocab, len(p4))
mplug_random = np.random.choice(mplug_vocab, len(p4))
# %% P4.1
p4_1, p4_2 = p4.copy(), p4.copy()
p5_0, p5_1, p5_2 = p5.copy(), p5.copy(), p5.copy()
p4["text"] = p4["text"].map(lambda x: x.split("? ")[0] + "? ")
p4_1["text"] = p4["text"] + ofa_random
p4_2["text"] = p4["text"] + ofa_random + " . " + mplug_random
p5["text"] = p5["text"].map(lambda x: x.split("hint: ")[0] + "hint: ")
p5_0["text"] = p5["text"]
p5_1["text"] = p5["text"] + ofa_random
p5_2["text"] = p5["text"] + ofa_random + " . " + mplug_random
# %%
FH.to_vg_csv(p4_1, f"{split}-P4.1")
FH.to_vg_csv(p4_2, f"{split}-P4.2")
# %%
FH.to_vg_csv(p5_0, f"{split}-P5.0")
FH.to_vg_csv(p5_1, f"{split}-P5.1")
FH.to_vg_csv(p5_2, f"{split}-P5.2")
