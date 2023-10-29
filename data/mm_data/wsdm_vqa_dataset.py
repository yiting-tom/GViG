import logging
import warnings
from uuid import uuid4

import numpy as np
import torch
from PIL import Image, ImageFile

import utils.transforms as T
from configs import paths as P
from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )

    patch_images = torch.stack([sample["patch_image"] for sample in samples], dim=0)
    patch_masks = torch.cat([sample["patch_mask"] for sample in samples])

    w_resize_ratios = torch.stack([s["w_resize_ratio"] for s in samples], dim=0)
    h_resize_ratios = torch.stack([s["h_resize_ratio"] for s in samples], dim=0)
    region_coords = torch.stack([s["region_coord"] for s in samples], dim=0)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        )
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens,
        },
        "target": target,
        "w_resize_ratios": w_resize_ratios,
        "h_resize_ratios": h_resize_ratios,
        "region_coords": region_coords,
    }

    return batch


class WSDMVQADataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=80,
        max_tgt_length=30,
        patch_image_size=512,
        imagenet_default_mean_and_std=False,
        num_bins=1000,
        max_image_size=512,
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.num_bins = num_bins

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        # for positioning
        self.positioning_transform = T.Compose(
            [
                T.RandomResize([patch_image_size], max_size=patch_image_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std, max_image_size=max_image_size),
            ]
        )

        if type(bpe).__name__ == "GPT2BPE":
            self.prompt = ' which region does the text " {} " describe?'
        elif type(bpe).__name__ == "BertBPE":
            self.prompt = '这段文字" {} "描述的是哪个区域？'

    def __getitem__(self, index):
        (image_url, width, height, x0, y0, x1, y1, text) = self.dataset[index]

        # read image and convert to RGB
        image = Image.open(P.WSDM_IMAGES_DIR / image_url.split("/")[-1]).convert("RGB")

        # change to float tensor
        width, height = int(width), int(height)
        region = torch.tensor(tuple(map(float, [x0, y0, x1, y1])))
        x0, y0, x1, y1 = region

        # generate box target
        boxes_target = {
            "boxes": region[None, :] * 1,
            "labels": np.array([0]),
            "area": torch.tensor([(x1 - x0) * (y1 - y0)]),
            "size": torch.tensor([height, width]),
        }

        # patch image
        patch_image, patch_boxes = self.positioning_transform(image, boxes_target)

        # resize ratio
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]

        # patch mask
        patch_mask = torch.tensor([True])

        # quantize box coordinates
        region_coord = "{} {} {} {} ".format(
            *(
                "<bin_{}>".format(
                    int((patch_boxes["boxes"][0][i] * (self.num_bins - 1)).round())
                )
                for i in range(4)
            )
        )

        src_caption = self.pre_caption(text, self.max_src_length)
        src_item = self.encode_text(self.prompt.format(src_caption))
        tgt_item = self.encode_text(region_coord, use_bpe=False)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        raise ValueError(self.bos_item, src_item, self.eos_item)
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": uuid4().int,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "w_resize_ratio": resize_w / width,
            "h_resize_ratio": resize_h / height,
            "region_coord": region,
        }
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
