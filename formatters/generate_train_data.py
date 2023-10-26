# %%
import pandas as pd
from wsdm_data.utils import img_formatter

from generate_wsdm_vg_data import RESULT_COLUMNS

# %%
train = pd.read_csv(
    "/home/P76104419/wsdm2023/VQA/dataset/2023/vg_input/final_train.csv",
    sep="\t",
    on_bad_lines="skip",
    names=RESULT_COLUMNS,
)
# %%
test = pd.read_csv(
    "/home/P76104419/dataset/wsdm/test_private/test_private.csv",
)
# %%
image_path = "/home/P76104419/dataset/wsdm/test_private/"
test["image"] = test["image"].apply(lambda x: image_path + x.split("/")[-1])
test["image"] = test["image"].map(img_formatter.filepath_to_base64)
# %%
test["bbox"] = test[["left", "top", "width", "height"]].agg(
    lambda x: ",".join(x.astype(str)), axis=1
)
# %%
test["text"] = test["question"]
test["unique_id"] = test.index
test["image_id"] = test.index
# %%
test.drop(
    columns=["width", "height", "left", "right", "top", "bottom", "question"],
    inplace=True,
    errors="ignore",
)
# %%
test[RESULT_COLUMNS].to_csv(
    "/home/P76104419/wsdm2023/VQA/dataset/iccv/test_private.tsv",
    sep="\t",
    index=False,
    header=False,
)
# %%
test


# %%
def format_to_vg_result_columns(df):
    df["text"] = df["text"](lambda x: x.split("?")[0] + "?")
    return df


# %%
train.to_csv(
    "dataset/iccv/only_q_train-77980.tsv",
    index=False,
    sep="\t",
    header=False,
)
