# Checkpoints Directory

This directory is designated for storing model checkpoint files generated during the training of models. Each checkpoint file represents the state of a model at a specific iteration or epoch, allowing for the restoration of training or evaluation of the model from a particular point.

## Structure

The typical structure of the `checkpoints` directory is as follows:

```bash
checkpoints/
│── [arch_1]
│  │── [exp_tag_1]
│  │  │── train-P[training_prompt_id_1]
│  │  │  │── val-P[validation_prompt_id_1]
│  │  │  │  │── checkpoint_[epoch_1]_[step_1].pt
│  │  │  │  ├── checkpoint_[epoch_1]_[step_2].pt
│  │  │  │  ├── checkpoint_[epoch_2]_[step_3].pt
│  │  │  │  ├── ...
│  │  │  │  ├── checkpoint_best_score_[best_score].pt
│  │  │  │  ├── checkpoint_last.pt
│  │  │  │  └── checkpoint_best.pt
│  │  │  └── val-P[validation_prompt_id_2]
│  │  │     │── checkpoint_[epoch_1]_[step_1].pt
│  │  │     │── ...
│  │  │     └── checkpoint_best.pt
│  │  └── train-P[training_prompt_id_2]
│  │     └── ...
│  └── [exp_tag_2]
│     └── ...
│── [arch_2]
│── ...
└── [arch_n]
```

## Description of Structure
- [arch_n]: Represents the architecture of the model. Different architectures have their separate directories.
- [exp_tag_n]: Represents the tag of the experiment. Each experiment conducted with different settings will have its own tag.
- train-P[training_prompt_id_n]: Represents the directory for a specific training prompt ID.
- val-P[validation_prompt_id_n]: Represents the directory for a specific validation prompt ID within the training prompt directory.
- checkpoint_[epoch_n]_[step_n].pt: Represents the checkpoint file saved at a specific epoch and step during training.
- checkpoint_best_score_[best_score].pt: Represents the checkpoint file with the best score achieved during training.
- checkpoint_last.pt: Represents the most recent checkpoint file saved.
- checkpoint_best.pt: Represents the checkpoint file with the best model performance on the validation set.

## Usage
The structure of the `checkpoints` directory is defined by the `scripts/train_single_exp.sh` or `scripts/train_multiple_exps.sh`. To modify the structure, you can make adjustments to these scripts according to your requirements.