# Logs Directory

This directory is designated for storing log files generated during the training of models. Each log file contains detailed information on the training process, including loss values, performance metrics, and debugging messages, which are essential for monitoring and analyzing model training.

## Structure

The typical structure of the `logs` directory is as follows:

```bash
logs/
│── [arch_1]
│  │── [exp_tag_1]
│  │  │── train-P[training_prompt_id_1]
│  │  │  │── val-P[validation_prompt_id_1].log
│  │  │  ├── ...
│  │  │  └── val-P[validation_prompt_id_n].log
│  │  └── train-P[training_prompt_id_2]
│  │     └── ...
│  └── [exp_tag_2]
│     └── ...
│── [arch_2]
│── ...
└── [arch_n]
```

## Description of Structure
[arch_n]: Represents the architecture of the model.
[exp_tag_n]: Represents the tag of the experiment.
train-P[training_prompt_id_n]: Represents the directory for a specific training prompt ID.
val-P[validation_prompt_id_n].log: Represents the log for training with a specific validation prompt ID.

## Usage
The structure of the logs directory is defined by the scripts/train_single_exp.sh or scripts/train_multiple_exps.sh. To modify the structure or log content, you can adjust these scripts according to your needs.
