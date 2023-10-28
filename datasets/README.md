# Datasets Directory

This is a designated space for you to place all the datasets you plan to use. 

## Structure

After downloading the WSDM VQA dataset, the typical structure of the `datasets` directory is as follows:

```bash
datasets/
├── images
├── official
├── README.md
├── [exp_tag_1]
│  │── test_private-P[prompt_name].csv
│  │── test_public-P[prompt_name].csv
│  └── train-P[prompt_name].csv
├── [exp_tag_2]
├── ...
└── [exp_tag_n]
   │── test_private-P[prompt_name].csv
   │── test_public-P[prompt_name].csv
   └── train-P[prompt_name].csv
```

## Description of Structure

- images: Represents the directory for the images of the WSDM VQA dataset.
- official: Represents the directory for the official splits of the WSDM VQA dataset.
- [exp_tag_n]: Represents the tag of the experiment. Each experiment conducted with different settings will have its own tag.

## Usage

The structure of the `datasets` directory is defined by the [`scripts/generate_prompt_data.sh`](../scripts/generate_prompt_data.sh). To modify the structure, you can make adjustments to these scripts according to your requirements.