import logging
from pathlib import Path

import pandas as pd

from configs import consts as C

L: logging.Logger = logging.getLogger(logging.basicConfig(level=logging.INFO))

PROMPT_TEMPLATES = {
    "Base": "{question}",
    "Base-1": "{question} {ofa_huge}",
    "Base-2": "{question} {ofa_huge} {ofa_large}",
    "Base-3": "{question} {ofa_huge} {ofa_large} {ofa_medium}",
    "Instruct-1": "question: {question} hint: {ofa_huge}",
    "Instruct-2": "question: {question} hint: {ofa_huge} . {ofa_large}",
    "Instruct-3": "question: {question} hint: {ofa_huge} . {ofa_large} . {ofa_medium}",
}


def generate_prompt(
    raw_file: str,
    answer_files: dict,
    prompt_name: str,
    output_file: str,
):
    logging.info(f"Generate prompt `{prompt_name}` to `{output_file}`")
    logging.info(f"Prompt template: `{PROMPT_TEMPLATES[prompt_name]}`")
    logging.info(f"Raw file: `{raw_file}`")
    for k, v in answer_files.items():
        logging.info(f"Answer file `{k}`: `{v}`")

    # Read raw file
    raw_df = pd.read_csv(raw_file)
    for k, v in answer_files.items():
        # Read answer file
        answer_df = pd.read_csv(v, names=C.VQA_PREDICT_COLUMNS)
        # Answer file should have the same length as raw file
        assert len(raw_df) == len(answer_df), f"{k} {len(raw_df)} != {len(answer_df)}"
        # Append answer column to raw file by given key
        raw_df[k] = answer_df["answer"]

    # Generate prompt
    assert prompt_name in PROMPT_TEMPLATES
    raw_df["question"] = raw_df.apply(
        lambda x: PROMPT_TEMPLATES[prompt_name].format(**x), axis=1
    )

    # Drop answer columns
    raw_df.drop(columns=list(answer_files.keys()), inplace=True)

    # Save prompt
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    raw_df.to_csv(output_file, index=False)
    logging.info(f"Save prompt to `{output_file}` ... Done")


def main_cli():
    """ Usage example:
    
    python prompt.py \
        --raw_file /GViG/datasets/official/train_sample.csv \
        --answer_files '{"ofa_huge":"/GViG/results/vqa/huge/official/train_sample.csv", "ofa_large":"/GViG/results/vqa/large/official/train_sample.csv"}' \
        --prompt_name Base \
        --output_file prompt.csv
    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_file", type=str, help="raw file")
    parser.add_argument(
        "--answer_files",
        type=str,
        help="a str of dict of answer files, e.g. {'ofa_huge': '/path/to/answer-huge.csv', 'ofa_large': '/path/to/answer-large.csv'}",
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        default=next(iter(PROMPT_TEMPLATES.keys())),
        choices=PROMPT_TEMPLATES.keys(),
        help="prompt name",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="default is: {raw_file}-P{prompt_name}.csv",
    )
    args = parser.parse_args()
    generate_prompt(
        args.raw_file,
        eval(args.answer_files),
        args.prompt_name,
        args.output_file
        if args.output_file
        else f"{args.raw_file.split('.')[0]}-P{args.prompt_name}.csv",
    )


if __name__ == "__main__":
    main_cli()
