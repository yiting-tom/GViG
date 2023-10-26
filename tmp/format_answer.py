from pathlib import Path

import pandas as pd

MODEL = "10_3e-5_512"

ROOT = Path(__file__).parent
RESULT_d = ROOT / "results"
SUBMIT_d = ROOT / "submit_answers" / MODEL

test_df = pd.read_csv(ROOT / "test_public.csv")
test_df["uniq_id"] = test_df.index


def format_and_zip(predict: pd.DataFrame, zip_path: Path) -> None:
    output = pd.merge(
        left=test_df,
        right=predict,
        left_on="uniq_id",
        right_index=True,
    )[["image", "box"]]
    output[["left", "top", "right", "bottom"]] = pd.DataFrame(
        list(output["box"].values)
    )
    del output["box"]

    def correct_ans(box, wh):
        box[box < 0] = 0
        left, top, right, bottom = box
        if left > right:
            right = wh[0]
        if top > bottom:
            bottom = wh[1]
        return dict(left=left, top=top, right=right, bottom=bottom)

    correct = []
    for i, d in output.iterrows():
        correct.append(
            correct_ans(
                box=d[["left", "top", "right", "bottom"]],
                wh=test_df[["width", "height"]].loc[i].values,
            )
        )
    out_ans = pd.DataFrame.from_dict(correct)
    out_file = pd.concat([test_df["image"], out_ans], axis=1)
    out_file.to_csv("answer.csv", index=False, compression="zip")
    Path("answer.csv").rename(zip_path)
    print(f"{zip_path} saved in {SUBMIT_d}")


if __name__ == "__main__":
    if not SUBMIT_d.exists():
        SUBMIT_d.mkdir(parents=True)
        print(f"{SUBMIT_d} created")

    for ckpt_d in (RESULT_d / MODEL).iterdir():
        assert ckpt_d.is_dir(), f"{ckpt_d} is not a directory"
        ckpt = ckpt_d.name
        zip_path = SUBMIT_d / f"{MODEL}-{ckpt}.zip"
        predicted_file = ckpt_d / "vg_predict.json"
        if zip_path.exists():
            continue
        predict = pd.read_json(predicted_file)
        format_and_zip(predict, zip_path)
