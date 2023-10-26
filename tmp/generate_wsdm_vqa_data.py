# %%
""" 
The information are separated by tabs.
    0. question-id
    1. image-id
    2. question
    3. answer (with confidence)
    4. predicted object labels (taken from VinVL, slightly brings around +0.1 accuracy improvement)
    5. image base64 string
e.g.)
79459	79459	is this person wearing shorts?	0.6|!+no	house&&short&&...&&sky	/9j/4AAQS...tigZ/9k=

"""
import argparse
import logging
from functools import partial
from pathlib import Path

import pandas as pd
from wsdm_data import utils

from configs import paths

L: logging.Logger = logging.getLogger(
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
)

VQA_DATASET_PATH_FMT = str(paths.GENERATED / "{split}.tsv")
RESULT_COLUMNS = ["question_id", "image_id", "question", "answer", "candidate", "image"]


# %%
def pipeline(d: pd.Series, image_dir: Path) -> dict:
    """pipeline

    Args:
        d (pd.Series): row of the dataframe
        image_dir (Path): directory of the images.

    Returns:
        pd.Series: processed row of the dataframe
    """
    image_filepath = utils.url_to_img_filepath(d["image"], image_dir)

    return dict(
        question=utils.reformat_question(d["question"]),
        image_id=utils.url_to_img_id(d["image"]),
        image=utils.filepath_to_base64(image_filepath),
    )


# %%
def generate_vqa_data(
    df: pd.DataFrame,
    image_dir: Path,
) -> pd.DataFrame:
    # update the columns
    df[list(set(RESULT_COLUMNS) - set(df.columns))] = 0
    # rename the columns
    df["image_url"] = df["image"]
    # apply image_dir into the pipeline
    part_pipeline = partial(pipeline, image_dir=image_dir)
    # apply the pipeline
    df.update(pd.DataFrame(df.apply(part_pipeline, axis=1).values.tolist()))
    # add question_id data by using the index
    df["question_id"] = df.index
    # add mock confidence and answer data
    df["answer"] = "0|!+foo"
    # add mock candidate data
    df["candidate"] = "house&&short"
    # return the result
    return df[RESULT_COLUMNS]


# %%


def main(split: str):
    # assign the tsv file path
    tsv_filepath = Path(VQA_DATASET_PATH_FMT.format(split=split))

    # check if the file exists
    if tsv_filepath.exists():
        L.info(f"{tsv_filepath} already exists - skipping")
        return None

    # map the split to the csv file path
    split_to_csv_file = {
        "train": paths.TRAIN_CSV,
        "train_sample": paths.TRAIN_SAMPLE_CSV,
        "test_public": paths.TEST_PUBLIC_CSV,
        "test_private": paths.TEST_PRIVATE_CSV,
    }

    # map the split to the image directory
    csv_to_image_dir = lambda x: x.parent / f"imgs-{str(x.stem)}"
    split_to_img_dir = {
        "train": csv_to_image_dir(paths.TRAIN_CSV),
        "train_sample": csv_to_image_dir(paths.TRAIN_SAMPLE_CSV),
        "test_public": csv_to_image_dir(paths.TEST_PUBLIC_CSV),
        "test_private": paths.TEST_PRIVATE_CSV.parent / "imgs",
    }

    # load the original csv data
    L.info(f"Loading original csv data: {split}")
    df = pd.read_csv(split_to_csv_file[split])

    # if the split is not test_private, download the images
    if split != "test_private":
        utils.download_images(df, split_to_img_dir[split])

    # generate the vqa data
    L.info(f"Generating VQA data: {split}")
    df = generate_vqa_data(
        df=df,
        image_dir=split_to_img_dir[split],
    )

    # save the data
    L.info(f"Saving VQA data: {split} to {tsv_filepath}")
    tsv_filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        tsv_filepath,
        sep="\t",
        index=False,
        header=False,
    )


# %%
if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test_public", "test_private", "train_sample"],
    )
    a = a.parse_args()
    main(a.split)
