# Results Directory

This directory is designated for storing the results generated during and after the training and evaluation of models. The results include model predictions, evaluation metrics, and any additional analysis or visualizations derived from the model's performance.

## Structure

The typical structure of the `results` directory is as follows:

```bash
results/
│── [arch_1]
│  │── [exp_tag_1]
│  │  │── train-P[training_prompt_id_1]
│  │  │  │── val-P[validation_prompt_id_1]
│  │  │  ├── ...
│  │  │  └── val-P[validation_prompt_id_2]
│  │  └── train-P[training_prompt_id_2]
│  │     └── ...
│  └── [exp_tag_2]
│     └── ...
│── [arch_2]
│── ...
└── [arch_n]
```

## Description of Structure

- [arch_n]: Represents the architecture of the model.
- [exp_tag_n]: Represents the tag of the experiment.
- train-P[training_prompt_id_n]: Represents the directory for a specific training prompt ID.
- val-P[validation_prompt_id_n]: Represents the result file for validating with a specific validation prompt ID.

## Usage

The structure of the results directory is defined by the scripts/train_single_exp.sh or scripts/train_multiple_exps.sh. To modify the structure or the content of the results, you can adjust these scripts according to your needs.