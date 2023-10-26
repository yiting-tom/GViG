# %%
import pandas as pd

from utils import file_handler as FH

# %%
df = pd.read_pickle("/home/P76104419/ICCV/dataset/base64/vg/test_private-entire.pkl")
question = df["text"].str.split("?").str[0] + "?"
for model_size in ["tiny", "medium", "base", "large"]:
    ofa = pd.read_json(
        f"/home/P76104419/ICCV/results/vqa/ofa_{model_size}/_predict.json"
    )
    ofa_2 = pd.read_json(
        f"/home/P76104419/ICCV/results/vqa/ofa_{model_size}-8/_predict.json"
    )
    P = {}
    P[6] = "question: " + question + " hint: " + ofa["answer"]

    for p in [6]:
        df["text"] = P[p]
        FH.to_vg_csv(df, f"small_vqa/{model_size}/test_private-P{p}")
# %%
