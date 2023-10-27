#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

from fairseq.data.dictionary import Dictionary

from .base_decoder import BaseDecoder
from .decoder_config import DecoderConfig, FlashlightDecoderConfig


def Decoder(
    cfg: Union[DecoderConfig, FlashlightDecoderConfig], tgt_dict: Dictionary
) -> BaseDecoder:
    if cfg.type == "viterbi":
        from .viterbi_decoder import ViterbiDecoder

        return ViterbiDecoder(tgt_dict)
    if cfg.type == "kenlm":
        from .flashlight_decoder import KenLMDecoder

        return KenLMDecoder(cfg, tgt_dict)
    if cfg.type == "fairseqlm":
        from .flashlight_decoder import FairseqLMDecoder

        return FairseqLMDecoder(cfg, tgt_dict)
    raise NotImplementedError(f"Invalid decoder name: {cfg.name}")
