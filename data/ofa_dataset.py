import logging
import re

import torch.utils.data

from fairseq.data import FairseqDataset

logger = logging.getLogger(__name__)


class OFADataset(FairseqDataset):
    def __init__(self, split, dataset, bpe, src_dict, tgt_dict):
        self.split = split
        self.dataset = dataset
        self.bpe = bpe
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.bos = src_dict.bos()
        self.eos = src_dict.eos()
        self.pad = src_dict.pad()
        self.bos_item = torch.LongTensor([self.bos])
        self.eos_item = torch.LongTensor([self.eos])

    def __len__(self):
        return len(self.dataset)

    def encode_text(
        self,
        text: str,
        length: int | None = None,
        append_bos: bool | None = False,
        append_eos: bool | None = False,
        use_bpe: bool | None = True,
    ) -> str:
        s = self.tgt_dict.encode_line(
            line=self.bpe.encode(text) if use_bpe else text,
            add_if_not_exist=False,
            append_eos=False,
        ).long()

        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s

    @staticmethod
    def pre_question(question: str, max_ques_words: int | None = None) -> str:
        question = (
            question.lower().lstrip(",.!?*#:;~").replace("-", " ").replace("/", " ")
        )

        question = re.sub(
            r"\s{2,}",
            " ",
            question,
        )
        question = question.rstrip("\n")
        question = question.strip(" ")

        # truncate question
        question_words = question.split(" ")
        if max_ques_words is not None and len(question_words) > max_ques_words:
            question = " ".join(question_words[:max_ques_words])

        return question

    @staticmethod
    def pre_caption(caption: str, max_words: int | None = None) -> str:
        caption = (
            caption.lower()
            .lstrip(",.!?*#:;~")
            .replace("-", " ")
            .replace("/", " ")
            .replace("<person>", "person")
        )

        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if max_words is not None and len(caption_words) > max_words:
            caption = " ".join(caption_words[:max_words])

        return caption
