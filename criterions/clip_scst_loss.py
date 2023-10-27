"""
criterions/clip_scst_loss.py

Copyright 2023 NCKU IKM Lab
Author: Yi-Ting Li 
Contact: yitingli.public@gmail.com
Description: This module contains a criterion implementation for a reward-driven approach to
train models using CLIP scores as rewards. It encompasses functionalities to convert tensor images to PIL,
calculate self-critical sequence training (SCST) loss, perform image-text similarity scoring using CLIP,
and compute overall loss with regards to generated and ground-truth samples. The module relies on Fairseq for
data processing, CLIP model from the `models` module for embeddings, and additional utilities for image and text
handling.

Functions:
    - custom_to_pil(x: torch.Tensor) -> Image.Image
        Converts a tensor image to PIL format.

    - scst_loss(lprobs: torch.Tensor, target: torch.Tensor, reward: torch.Tensor, 
                ignore_index: Optional[int], reduce: bool) -> Tuple[torch.Tensor, int]
        Computes the SCST loss.

Classes:
    - ClipScstRewardCriterionConfig
        Dataclass for storing the configuration of ClipScstRewardCriterion.

    - ClipScstRewardCriterion(FairseqCriterion)
        Implementation of the FairseqCriterion for computing the training loss based on CLIP rewards.
"""
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
from omegaconf import II
from PIL import Image

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import data_utils
from fairseq.dataclass import FairseqDataclass
from models import clip


def custom_to_pil(x: torch.Tensor) -> Image.Image:
    """
    Converts a tensor image to a PIL Image.

    Args:
        x (torch.Tensor): The image tensor to be converted.

    Returns:
        Image.Image: The converted PIL Image.
    """
    x = x.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x: Image.Image = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def scst_loss(
    lprobs: torch.Tensor,
    target: torch.Tensor,
    reward: torch.Tensor,
    ignore_index: Optional[int] = None,
    reduce: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Computes the Self-Critical Sequence Training (SCST) loss.

    Args:
        lprobs (torch.Tensor): Log probabilities of the predicted sequence.
        target (torch.Tensor): The ground truth sequence.
        reward (torch.Tensor): The reward obtained for each token in the sequence.
        ignore_index (Optional[int]): The index to be ignored in the loss computation. Default is None.
        reduce (bool): Whether to reduce the loss over the batch. Default is True.

    Returns:
        Tuple[torch.Tensor, int]: The computed SCST loss and the number of tokens.
    """
    loss = -lprobs.gather(
        dim=-1, index=target.unsqueeze(-1)
    ).squeeze() * reward.unsqueeze(-1)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        loss.masked_fill_(pad_mask, 0.0)
        ntokens = (~pad_mask).sum()
    else:
        loss = loss.squeeze(-1)
        ntokens = target.numel()
    if reduce:
        loss = loss.sum()
    return loss, ntokens


@dataclass
class ClipScstRewardCriterionConfig(FairseqDataclass):
    """
    Configuration Dataclass for ClipScstRewardCriterion.

    Attributes:
        ignore_prefix_size (int): Number of tokens to ignore from the beginning of each sequence.
        sentence_avg (bool): Whether to average the loss over sentences.
        constraint_range (Optional[str]): The constraint range for tokens. Default is None.
    """

    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    constraint_range: Optional[str] = field(
        default=None, metadata={"help": "constraint range"}
    )


@register_criterion(
    "clip_scst_reward_criterion", dataclass=ClipScstRewardCriterionConfig
)
class ClipScstRewardCriterion(FairseqCriterion):
    """
    Criterion for Sequence-level knowledge distillation based on CLIP rewards.

    This criterion is designed to optimize models using rewards obtained from a CLIP model.
    In this approach, the training loss is derived from the similarities between the generated
    output images and the ground-truth textual descriptions. The similarities are measured using
    a CLIP model, which jointly embeds images and text in a shared space.

    Attributes:
        sentence_avg (bool): If True, averages the loss over sentences, else over tokens.
        ignore_prefix_size (int): Number of tokens to be ignored from the start of each sequence.
        constraint_start (Optional[int]): Start index of constraint range for tokens.
        constraint_end (Optional[int]): End index of constraint range for tokens.
        CLIP_REWARD_WEIGHT (float): Weight for the CLIP reward when computing the loss.

    Methods:
        forward: Computes the loss for the given sample and returns loss, sample size, and logging outputs.
        _calculate_clip_scores: Calculates the similarity scores between generated images and ground truth text using CLIP.
        get_generator_out: Gets the generated images, their corresponding tokens, and the ground truth text for the sample.
        get_reward_and_scores: Computes the rewards and scores for the generated images based on CLIP similarity.
        get_net_output: Computes the network's outputs for the given sample and generated targets.
        get_lprobs_and_target: Gets the log probabilities and target values after applying constraints and ignore prefixes.
        compute_loss: Main method to compute the loss based on CLIP rewards.
        reduce_metrics: Aggregates logging outputs from data parallel training.
        logging_outputs_can_be_summed: Indicates if logging outputs from `forward` can be summed across workers.
    """

    CLIP_REWARD_WEIGHT = 2.5

    def __init__(self, task, sentence_avg, ignore_prefix_size=0, constraint_range=None):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.ignore_prefix_size = ignore_prefix_size

        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(",")
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)

    def forward(self, model, sample, reduce=True):
        """
        Computes the forward pass for the given model and sample.

        Args:
            model: The model object to compute the forward pass.
            sample: The input data sample.
            reduce (bool, optional): Whether to reduce the loss over batches. Default is True.

        Returns:
            tuple: A tuple containing the loss, sample size, and logging outputs.
        """

        loss, score, ntokens, nsentences = self.compute_loss(
            model, sample, reduce=reduce
        )

        sample_size = nsentences if self.sentence_avg else ntokens
        logging_output = {
            "loss": loss.data,
            "score": score,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def _calculate_clip_scores(self, gen_res, gt_text, device):
        """
        Calculates the clip scores for generated images and ground truth text.

        Args:
            gen_res: A list of generated images.
            gt_text: The input ground truth captions.
            device: The device for the clip model.

        Returns:
            torch.Tensor: The calculated scores.
        """

        batch_size = len(gt_text)
        gen_res_size = len(gen_res)
        img_per_seq = gen_res_size // batch_size

        hyp_images = torch.stack(
            [self.task.clip_preprocess(gen_image) for gen_image in gen_res], dim=0
        ).to(device)

        clip_input = clip.tokenize([text for text in gt_text]).to(device)
        with torch.no_grad():
            image_features = self.task.clip_model.encode_image(hyp_images)
            text_features = self.task.clip_model.encode_text(clip_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features = image_features.view(batch_size, img_per_seq, -1)
            text_features = text_features.view(batch_size, 1, -1)
            ti_similarity = image_features @ text_features.transpose(1, 2)
            ti_similarity = ti_similarity.view(-1)

        scores = self.CLIP_REWARD_WEIGHT * ti_similarity
        return scores

    def get_generator_out(self, model, sample):
        """
        Generates output using the model and the given sample.

        Args:
            model: The model object used for generating the output.
            sample: The input data sample.

        Returns:
            tuple: A tuple containing the generated target, generated result, and ground truth text.
        """

        model.eval()
        with torch.no_grad():
            self.task.scst_generator.model.eval()
            gen_out = self.task.scst_generator.generate([model], sample)

        gen_target = []
        gen_res = []
        gt_text = []
        for i in range(len(gen_out)):
            with torch.no_grad():
                tokens = torch.stack(
                    [item["tokens"][:-1] for item in gen_out[i]], dim=0
                )
                tokens += (
                    -len(self.task.src_dict)
                    + self.task.cfg.code_dict_size
                    + self.task.cfg.num_bins
                )
                images = self.task.image_tokenizer.decode_code(
                    tokens.view(
                        -1,
                        self.task.cfg.code_image_size // 8,
                        self.task.cfg.code_image_size // 8,
                    )
                )
                images = [custom_to_pil(image) for image in images]

            gen_target += [item["tokens"] for item in gen_out[i]]
            gen_res += images
            gt_text.append(
                self.task.bpe.decode(
                    self.task.tgt_dict.string(
                        utils.strip_pad(
                            sample["net_input"]["src_tokens"][i], self.padding_idx
                        )
                        .cpu()
                        .int()
                    )
                )[
                    38:
                ]  # remove task instruction.
            )

        return gen_target, gen_res, gt_text

    def get_reward_and_scores(self, gen_res, gt_text, device):
        """
        Computes the reward and scores for generated results and ground truth text.

        Args:
            gen_res: A list of generated results.
            gt_text: The input ground truth captions.
            device: The device for calculations.

        Returns:
            tuple: A tuple containing the rewards and scores.
        """

        batch_size = len(gt_text)
        gen_res_size = len(gen_res)
        img_per_sample = gen_res_size // batch_size

        scores = self._calculate_clip_scores(gen_res, gt_text, device)
        sc_ = scores.reshape(batch_size, img_per_sample)
        baseline = (sc_.sum(1, keepdim=True) - sc_) / (sc_.shape[1] - 1)
        # sample - baseline
        reward = scores.reshape(batch_size, img_per_sample)
        reward = reward - baseline
        reward = reward.view(-1)

        return reward, scores

    def get_net_output(self, model, sample, gen_target):
        """
        Gets the network output for the given model, sample, and generated target.

        Args:
            model: The model object.
            sample: The input data sample.
            gen_target: The generated target data.

        Returns:
            tuple: A tuple containing the network output and generated target tokens.
        """

        def merge(
            sample_list, eos=self.task.tgt_dict.eos(), move_eos_to_beginning=False
        ):
            return data_utils.collate_tokens(
                sample_list,
                pad_idx=self.padding_idx,
                eos_idx=eos,
                left_pad=False,
                move_eos_to_beginning=move_eos_to_beginning,
            )

        batch_size = len(sample["target"])
        gen_target_size = len(gen_target)
        img_per_sample = gen_target_size // batch_size

        model.train()
        sample_src_tokens = torch.repeat_interleave(
            sample["net_input"]["src_tokens"], img_per_sample, dim=0
        )
        sample_src_lengths = torch.repeat_interleave(
            sample["net_input"]["src_lengths"], img_per_sample, dim=0
        )
        sample_code_masks = torch.repeat_interleave(
            sample["net_input"]["code_masks"], img_per_sample, dim=0
        )
        gen_prev_output_tokens = torch.as_tensor(
            merge(gen_target, eos=self.task.tgt_dict.bos(), move_eos_to_beginning=True),
            device=sample["target"].device,
            dtype=torch.int64,
        )
        gen_target_tokens = torch.as_tensor(
            merge(gen_target), device=sample["target"].device, dtype=torch.int64
        )
        net_output = model(
            src_tokens=sample_src_tokens,
            src_lengths=sample_src_lengths,
            code_masks=sample_code_masks,
            prev_output_tokens=gen_prev_output_tokens,
        )

        return net_output, gen_target_tokens

    def get_lprobs_and_target(self, model, net_output, gen_target):
        """
        Gets the log probabilities and target for the given model, network output, and generated target.

        Args:
            model: The model object.
            net_output: The network output.
            gen_target: The generated target data.

        Returns:
            tuple: A tuple containing the log probabilities and generated target tokens.
        """

        if self.constraint_start is not None and self.constraint_end is not None:
            net_output[0][:, :, 4 : self.constraint_start] = -math.inf
            net_output[0][:, :, self.constraint_end :] = -math.inf
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                gen_target = gen_target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                gen_target = gen_target[self.ignore_prefix_size :, :].contiguous()
        return lprobs, gen_target

    def compute_loss(self, model, sample, reduce=True):
        """
        Computes the loss for the given model and sample.

        Args:
            model: The model object.
            sample: The input data sample.
            reduce (bool, optional): Whether to reduce the loss over batches. Default is True.

        Returns:
            tuple: A tuple containing the loss, total score, number of tokens, and number of sentences.
        """

        gen_target, gen_res, gt_text = self.get_generator_out(model, sample)
        reward, scores = self.get_reward_and_scores(
            gen_res, gt_text, device=sample["target"].device
        )
        net_output, gen_target_tokens = self.get_net_output(model, sample, gen_target)
        gen_lprobs, gen_target_tokens = self.get_lprobs_and_target(
            model, net_output, gen_target_tokens
        )
        loss, ntokens = scst_loss(
            gen_lprobs,
            gen_target_tokens,
            reward,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        nsentences = gen_target_tokens.size(0)

        return loss, scores.sum(), ntokens, nsentences

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """
        Aggregates logging outputs from data parallel training.

        Args:
            logging_outputs: A list of dictionaries containing logging outputs.
        """

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        score_sum = sum(log.get("score", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("score", score_sum / nsentences, nsentences, round=3)

        metrics.log_scalar("ntokens", ntokens, 1, round=3)
        metrics.log_scalar("nsentences", nsentences, 1, round=3)
        metrics.log_scalar("sample_size", sample_size, 1, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Checks whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`.

        Returns:
            bool: True if the logging outputs can be summed, False otherwise.
        """

        return True
