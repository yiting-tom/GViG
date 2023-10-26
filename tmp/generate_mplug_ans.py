# %%
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=60)
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
from tqdm import tqdm

from configs import paths as P
from utils import file_handler as FH
from utils import img_fmt as IF

# %%
MODEL_ID = "damo/mplug_visual-question-answering_coco_base_en"
# df = pd.read_pickle(P.VG_64 / 'gqa_train.pkl')
# %%
idx = len(df) // 4
df = df.iloc[idx * 3 :]
# %%
images = df["image"].parallel_map(np.array)
# %%
input_pairs = list(zip(images, df["text"]))
# %%
pd.DataFrame(input_pairs).to_pickle("./gqa-mplug-train-input-4.pkl")
# %%
pipeline_vqa = pipeline(
    task=Tasks.visual_question_answering,
    model=MODEL_ID,
    device="gpu",
)
# %%
mplug_predict = pipeline_vqa(input_pairs)

pd.to_pickle(pd.DataFrame(mplug_predict), "./gqa-mplug-train-1.pkl")
# %%

# %%
# for idx in range(1,2):
df = pd.read_pickle(P.VQA_64 / f"gqa-mplug-train-input-{idx}.pkl")
df.columns = ["image", "text"]
# df.to_feather(P.VQA_64 / f'gqa-mplug-train-input-{idx}.feather')
# %%
import numpy as np

idx = 1
np.save(P.VQA_64 / f"gqa-mplug-train-input-{idx}.pkl", df.to_dict("records"))
# %%
out = np.load(P.VQA_64 / f"gqa-mplug-train-input-{idx}.pkl", allow_pickle=True)
# %%
images = df["image"].parallel_map(np.array)
# %%
df = pd.read_pickle(
    "/home/P76104419/ICCV/dataset/base64/vqa/gqa-mplug-train-input-1.pkl"
)
# %%
df
