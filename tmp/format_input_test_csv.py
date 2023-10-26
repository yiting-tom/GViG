import logging
import re
import unicodedata

import pandas as pd

from configs import paths

L: logging.Logger = logging.getLogger(logging.basicConfig(level=logging.INFO))


def full_width_to_half_width(s: str) -> str:
    """full-width to half-width and translate to ascii with unicodedata"""
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )


def translate_head_punctuations(s: str) -> str:
    """translate head punctuations to specific format"""
    sp = s.split(" ")
    x = sp[0]
    x = re.sub(r"w[wh]h", "wh", x)
    x = re.sub(r"\'s", " is", x)
    x = re.sub(r"\'re", " are", x)
    x = re.sub(r"\'|\"|-|_|\,|/|\.", "", x)
    sp[0] = x
    result = " ".join(sp).lower()
    return result.lower()


def reformat_question(q: str) -> str:
    """reformat question to specific format"""
    q = q.lower()
    q = full_width_to_half_width(q)
    q = translate_head_punctuations(q)
    if not q.endswith("?"):
        q += "?"
    while q.endswith("??"):
        q = q[:-1]
    if len(q.split("?")) > 2:
        q = q.split("?")[0] + "?"
    q = q.capitalize()

    return q


def main():
    test_csv = "dataset/train.csv"
    formatted_test_csv = "dataset/formatted_train.pkl"

    L.info(f"read test csv: {test_csv}")
    df = pd.read_csv(test_csv)

    L.info(f"reformat question")
    df["question"] = df["question"].apply(reformat_question)

    L.info(f"save test csv: {formatted_test_csv}")
    df.to_pickle(formatted_test_csv)


if __name__ == "__main__":
    main()
