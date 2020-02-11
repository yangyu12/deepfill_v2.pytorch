import logging
import copy
import numpy as np
from fvcore.common.file_io import PathManager
import os
from PIL import Image, ImageDraw
import math

import torch.utils.data

from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog
)
from detectron2.data.datasets.pascal_voc import CLASS_NAMES
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T


class InpainterDatasetMapper:
    def __init__(self, cfg, is_train=True):
        # build_transform_gen
        if is_train:
            max_size = cfg.INPUT.MAX_SIZE_TRAIN  # currently only square shape
        else:
            # TODO: multi-scale test
            max_size = cfg.INPUT.MAX_SIZE_TEST
        logger = logging.getLogger(__name__)
        tfm_gens = []
        tfm_gens.append(T.Resize(max_size))
        if is_train:
            tfm_gens.append(T.RandomFlip())
            logger.info("TransformGens used in training: " + str(tfm_gens))
        else:
            logger.info("TransformGens used in test: " + str(tfm_gens))
        self.tfm_gens = tfm_gens

        # random crop augmentation
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.img_format         = cfg.INPUT.FORMAT
        self.is_train           = is_train
        self.mask_type          = cfg.INPUT.TRAIN_MASK_TYPE
        self.max_shift          = cfg.INPUT.MAX_SHIFT

        # TODO: configure it
        self.min_num_vertex     = 4
        self.max_num_vertex     = 12
        self.mean_angle         = 2 * math.pi / 5  # 72 deg
        self.angle_range        = 2 * math.pi / 15  # 24 deg
        self.min_width          = 12
        self.max_width          = 40

        assert self.mask_type in ["random_regular", "random_irregular"]

    def __call__(self, dataset_dict):
        """
        :param dataset_dict:
        :return:
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # first resize, then crop
        if self.is_train:
            assert "annotations" not in dataset_dict
            image, transforms = T.apply_transform_gens(
                self.tfm_gens + ([self.crop_gen] if self.crop_gen else []), image
            )

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day

        if not self.is_train:
            mask = np.load(dataset_dict["mask_file_name"])
            # mask = transforms.apply_segmentation(mask)
            mask = torch.as_tensor(mask.astype(np.float32)[None])  # (1, H, W)
            dataset_dict["mask"] = mask
            return dataset_dict

        # Option 1: randomly generate brush stroke as Yu et al. 2019
        if self.mask_type == "random_regular":
            raise NotImplementedError
        elif self.mask_type == "random_irregular":
            mask = self.generate_random_stroke_mask(image.shape[:2])
        else:
            raise ValueError(f"Unexpected mask type, got {self.mask_type}")
        mask = torch.as_tensor(mask.astype(np.float32)[None])  # (1, H, W)
        dataset_dict["mask"] = mask
        return dataset_dict

    def generate_random_stroke_mask(self, spatial_size):
        H, W = spatial_size
        mask = Image.new('L', (W, H), 0)
        average_radius = math.sqrt(H * H + W * W) / 8

        # randomly draw some strokes
        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(self.min_num_vertex, self.max_num_vertex)
            angle_min = self.mean_angle - np.random.uniform(0, self.angle_range)  #
            angle_max = self.mean_angle + np.random.uniform(0, self.angle_range)
            angles = []
            vertex = []
            # produce z shape angles
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            # produce vertex based on initial vertex, random length, and relative angles
            vertex.append((int(np.random.randint(0, W)), int(np.random.randint(0, H))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius // 2),
                    0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, W-1)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, H-1)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(self.min_width, self.max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        return mask


# =====================================================
# TODO: register dataset for training and evaluation
#
def load_places2_inpainting_data(dirname: str):
    # TODO:
    fileids = []

    dicts = []
    for fileid in fileids:
        # TODO:
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
        mask_file = os.path.join(dirname, "ShiftMasks", fileid + ".npy")

        with PathManager.open(jpeg_file, "rb") as f:
            im = Image.open(f)
            im = np.asarray(im, dtype="uint8")

        # TODO:

        r = {
            "file_name": jpeg_file,
            "mask_file_name": mask_file,
            "image_id": fileid,
            "height": im.shape[0],
            "width": im.shape[1],
        }
        dicts.append(r)
    return dicts


def register_places2_inpainting_data(name, dirname):
    DatasetCatalog.register(name, lambda: load_places2_inpainting_data(dirname))
    MetadataCatalog.get(name).set(dirname=dirname)

# TODO:
# register_places2_inpainting_data('places2_train', './datasets/places2')
# register_places2_inpainting_data('places2_test', './datasets/places2')

