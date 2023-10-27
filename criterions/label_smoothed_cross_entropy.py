"""
criterions/label_smoothed_cross_entropy.py

Copyright 2023 NCKU IKM Lab
Author: Yi-Ting Li 
Contact: yitingli.public@gmail.com
Description: This script contains the implementation of a criterion for label-smoothed cross-entropy loss.
The AdjustLabelSmoothedCrossEntropyCriterion class is designed to be used with the Fairseq library and supports various
configurations and settings such as label smoothing, ignoring prefixes, reporting accuracy, sample patching, constraint ranges,
and regularization with R-Drop. The script also includes utility functions and methods for computing losses, handling 
R-Drop samples, and calculating KL divergence.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import II

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class AdjustLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    """
    A configuration class for the AdjustLabelSmoothedCrossEntropyCriterion.

    Attributes:
        label_smoothing (float): Epsilon for label smoothing. 0 means no label smoothing.
        report_accuracy (bool): If True, report accuracy metric.
        ignore_prefix_size (int): Number of first tokens to ignore.
        ignore_eos (bool): If True, Ignore end-of-sequence (eos) token.
        sentence_avg (bool): If True, sentence-level average is used in optimization.
        drop_worst_ratio (float): Ratio for discarding bad samples.
        drop_worst_after (int): Number of steps after which bad samples are discarded.
        use_rdrop (bool): If True, use R-Drop regularization technique.
        reg_alpha (float): Weight for R-Drop regularization.
        sample_patch_num (int): Number of sample patches for version 1.
        constraint_range (Optional[str]): Range of constraint, if any.
    """

    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    ignore_eos: bool = field(
        default=False,
        metadata={"help": "Ignore eos token"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    drop_worst_ratio: float = field(
        default=0.0,
        metadata={"help": "ratio for discarding bad samples"},
    )
    drop_worst_after: int = field(
        default=0,
        metadata={"help": "steps for discarding bad samples"},
    )
    use_rdrop: bool = field(default=False, metadata={"help": "use R-Drop"})
    reg_alpha: float = field(default=1.0, metadata={"help": "weight for R-Drop"})
    sample_patch_num: int = field(
        default=196, metadata={"help": "sample patches for v1"}
    )
    constraint_range: Optional[str] = field(
        default=None, metadata={"help": "constraint range"}
    )


def construct_rdrop_sample(
    x: Union[Dict, torch.Tensor, int, np.ndarray]
) -> Union[Dict, torch.Tensor, int, np.ndarray]:
    """
    Constructs a sample for R-Drop by duplicating the input sample.

    Args:
        x (Union[dict, torch.Tensor, int, np.ndarray]): Input sample.

    Returns:
        Union[dict, torch.Tensor, int, np.ndarray]: Duplicated input sample.

    Raises:
        NotImplementedError: If the input type is not supported.
    """

    if isinstance(x, dict):
        for key in x:
            x[key] = construct_rdrop_sample(x[key])
        return x
    elif isinstance(x, torch.Tensor):
        return x.repeat(2, *([1] * (x.dim() - 1)))
    elif isinstance(x, int):
        return x * 2
    elif isinstance(x, np.ndarray):
        return x.repeat(2)
    else:
        raise NotImplementedError


def kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Computes the symmetrized KL divergence loss between two distributions.

    Args:
        p (torch.Tensor): The first distribution.
        q (torch.Tensor): The second distribution.

    Returns:
        torch.Tensor: The computed KL divergence loss.
    """

    p_loss = F.kl_div(p, torch.exp(q), reduction="sum")
    q_loss = F.kl_div(q, torch.exp(p), reduction="sum")
    loss = (p_loss + q_loss) / 2
    return loss


def label_smoothed_nll_loss(
    lprobs: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    update_num: int,
    drop_worst_ratio: float = 0.0,
    drop_worst_after: int = 0,
    use_rdrop: bool = False,
    reg_alpha: float = 1.0,
    reduce: bool = True,
    constraint_masks: Optional[torch.Tensor] = None,
    constraint_start: Optional[int] = None,
    constraint_end: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Computes the label-smoothed negative log-likelihood loss.

    Args:
        lprobs (torch.Tensor): Log probabilities of predictions.
        target (torch.Tensor): Ground truth labels.
        epsilon (float): Smoothing factor for label smoothing.
        update_num (int): Current update number.
        drop_worst_ratio (float, optional): Ratio for discarding worst samples. Default is 0.0.
        drop_worst_after (int, optional): Number of updates after which to drop worst samples. Default is 0.
        use_rdrop (bool, optional): Whether to use R-Drop. Default is False.
        reg_alpha (float, optional): Regularization weight for R-Drop. Default is 1.0.
        reduce (bool, optional): Whether to reduce the loss. Default is True.
        constraint_masks (Optional[torch.Tensor], optional): Constraint masks for label smoothing. Default is None.
        constraint_start (Optional[int], optional): Start index for constraint range. Default is None.
        constraint_end (Optional[int], optional): End index for constraint range. Default is None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, int]: The total loss, the NLL loss, and the number of tokens.
    """

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)
    if constraint_masks is not None:
        smooth_loss = (
            -lprobs.masked_fill(~constraint_masks, 0)
            .sum(dim=-1, keepdim=True)
            .squeeze(-1)
        )
        eps_i = epsilon / (constraint_masks.sum(1) - 1 + 1e-6)
    elif constraint_start is not None and constraint_end is not None:
        constraint_range = [0, 1, 2, 3] + list(range(constraint_start, constraint_end))
        smooth_loss = -lprobs[:, constraint_range].sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (len(constraint_range) - 1 + 1e-6)
    else:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    if drop_worst_ratio > 0 and update_num > drop_worst_after:
        if use_rdrop:
            true_batch_size = loss.size(0) // 2
            _, indices = torch.topk(
                loss[:true_batch_size],
                k=int(true_batch_size * (1 - drop_worst_ratio)),
                largest=False,
            )
            loss = torch.cat([loss[indices], loss[indices + true_batch_size]])
            nll_loss = torch.cat(
                [nll_loss[indices], nll_loss[indices + true_batch_size]]
            )
            lprobs = torch.cat([lprobs[indices], lprobs[indices + true_batch_size]])
        else:
            loss, indices = torch.topk(
                loss, k=int(loss.shape[0] * (1 - drop_worst_ratio)), largest=False
            )
            nll_loss = nll_loss[indices]
            lprobs = lprobs[indices]

    ntokens = loss.numel()
    nll_loss = nll_loss.sum()
    loss = loss.sum()
    if use_rdrop:
        true_batch_size = lprobs.size(0) // 2
        p = lprobs[:true_batch_size]
        q = lprobs[true_batch_size:]
        if constraint_start is not None and constraint_end is not None:
            constraint_range = [0, 1, 2, 3] + list(
                range(constraint_start, constraint_end)
            )
            p = p[:, constraint_range]
            q = q[:, constraint_range]
        loss += kl_loss(p, q) * reg_alpha

    return loss, nll_loss, ntokens


@register_criterion(
    "adjust_label_smoothed_cross_entropy",
    dataclass=AdjustLabelSmoothedCrossEntropyCriterionConfig,
)
class AdjustLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task: Any,
        sentence_avg: bool,
        label_smoothing: float,
        ignore_prefix_size: int = 0,
        ignore_eos: bool = False,
        report_accuracy: bool = False,
        drop_worst_ratio: float = 0,
        drop_worst_after: int = 0,
        use_rdrop: bool = False,
        reg_alpha: float = 1.0,
        sample_patch_num: int = 196,
        constraint_range: Optional[str] = None,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.ignore_eos = ignore_eos
        self.report_accuracy = report_accuracy
        self.drop_worst_ratio = drop_worst_ratio
        self.drop_worst_after = drop_worst_after
        self.use_rdrop = use_rdrop
        self.reg_alpha = reg_alpha
        self.sample_patch_num = sample_patch_num

        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(",")
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)

    def forward(
        self,
        model: Any,
        sample: Union[Dict[str, Any], List[Dict[str, Any]]],
        update_num: int = 0,
        reduce: bool = True,
    ) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if isinstance(sample, list):
            if self.sample_patch_num > 0:
                sample[0]["net_input"]["sample_patch_num"] = self.sample_patch_num
            loss_v1, sample_size_v1, logging_output_v1 = self.forward(
                model, sample[0], update_num, reduce
            )
            loss_v2, sample_size_v2, logging_output_v2 = self.forward(
                model, sample[1], update_num, reduce
            )
            loss = loss_v1 / sample_size_v1 + loss_v2 / sample_size_v2
            sample_size = 1
            logging_output = {
                "loss": loss.data,
                "loss_v1": loss_v1.data,
                "loss_v2": loss_v2.data,
                "nll_loss": logging_output_v1["nll_loss"].data / sample_size_v1
                + logging_output_v2["nll_loss"].data / sample_size_v2,
                "ntokens": logging_output_v1["ntokens"] + logging_output_v2["ntokens"],
                "nsentences": logging_output_v1["nsentences"]
                + logging_output_v2["nsentences"],
                "sample_size": 1,
                "sample_size_v1": sample_size_v1,
                "sample_size_v2": sample_size_v2,
            }
            return loss, sample_size, logging_output

        if self.use_rdrop:
            construct_rdrop_sample(sample)

        net_output = model(**sample["net_input"])
        loss, nll_loss, ntokens = self.compute_loss(
            model, net_output, sample, update_num, reduce=reduce
        )
        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(
        self, model: Any, net_output: Tuple[torch.Tensor], sample: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        conf = (
            sample["conf"][:, None, None]
            if "conf" in sample and sample["conf"] is not None
            else 1
        )
        constraint_masks = None
        if "constraint_masks" in sample and sample["constraint_masks"] is not None:
            constraint_masks = sample["constraint_masks"]
            net_output[0].masked_fill_(~constraint_masks, -math.inf)
        if self.constraint_start is not None and self.constraint_end is not None:
            net_output[0][:, :, 4 : self.constraint_start] = -math.inf
            net_output[0][:, :, self.constraint_end :] = -math.inf
        lprobs = model.get_normalized_probs(net_output, log_probs=True) * conf
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
            if constraint_masks is not None:
                constraint_masks = constraint_masks[
                    :, self.ignore_prefix_size :, :
                ].contiguous()
        if self.ignore_eos:
            bsz, seq_len, embed_dim = lprobs.size()
            eos_indices = target.eq(self.task.tgt_dict.eos())
            lprobs = lprobs[~eos_indices].reshape(bsz, seq_len - 1, embed_dim)
            target = target[~eos_indices].reshape(bsz, seq_len - 1)
            if constraint_masks is not None:
                constraint_masks = constraint_masks[~eos_indices].reshape(
                    bsz, seq_len - 1, embed_dim
                )
        if constraint_masks is not None:
            constraint_masks = constraint_masks.view(-1, constraint_masks.size(-1))
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1), constraint_masks

    def compute_loss(
        self,
        model: Any,
        net_output: Tuple[torch.Tensor],
        sample: Dict[str, Any],
        update_num: int,
        reduce: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        lprobs, target, constraint_masks = self.get_lprobs_and_target(
            model, net_output, sample
        )
        if constraint_masks is not None:
            constraint_masks = constraint_masks[target != self.padding_idx]
        lprobs = lprobs[target != self.padding_idx]
        target = target[target != self.padding_idx]
        loss, nll_loss, ntokens = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            update_num,
            reduce=reduce,
            drop_worst_ratio=self.drop_worst_ratio,
            drop_worst_after=self.drop_worst_after,
            use_rdrop=self.use_rdrop,
            reg_alpha=self.reg_alpha,
            constraint_masks=constraint_masks,
            constraint_start=self.constraint_start,
            constraint_end=self.constraint_end,
        )
        return loss, nll_loss, ntokens

    def compute_accuracy(
        self, model: Any, net_output: Tuple[torch.Tensor], sample: Dict[str, Any]
    ) -> Tuple[int, int]:
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_sum_v1 = sum(log.get("loss_v1", 0) for log in logging_outputs)
        loss_sum_v2 = sum(log.get("loss_v2", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        sample_size_v1 = sum(log.get("sample_size_v1", 0) for log in logging_outputs)
        sample_size_v2 = sum(log.get("sample_size_v2", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar(
            "loss_v1",
            loss_sum_v1 / max(sample_size_v1, 1),
            max(sample_size_v1, 1),
            round=3,
        )
        metrics.log_scalar(
            "loss_v2",
            loss_sum_v2 / max(sample_size_v2, 1),
            max(sample_size_v2, 1),
            round=3,
        )
        metrics.log_scalar("nll_loss", nll_loss_sum / sample_size, ntokens, round=3)
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        metrics.log_scalar("ntokens", ntokens, 1, round=3)
        metrics.log_scalar("nsentences", nsentences, 1, round=3)
        metrics.log_scalar("sample_size", sample_size, 1, round=3)
        metrics.log_scalar("sample_size_v1", sample_size_v1, 1, round=3)
        metrics.log_scalar("sample_size_v2", sample_size_v2, 1, round=3)

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
