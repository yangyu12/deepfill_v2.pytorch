# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.evaluation import DatasetEvaluator


class InpaintingEvaluator(DatasetEvaluator):
    """
    Evaluate image inpainting
    """
    def __init__(self, dataset_name, output_dir=None):
        self._dataset_name = dataset_name
        self._im_val_range = [0, 1]
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._rel_l1_err = OrderedDict()
        self._rel_l2_err = OrderedDict()

    def process(self, inputs, outputs):
        # TODO: maybe this evaluation is wrong
        for input, output in zip(inputs, outputs):
            output = output["inpainted"].to(self._cpu_device)
            pred_im = np.array(output, dtype=np.float32)
            pred_im = np.transpose((pred_im + 1.) / 2., (1, 2, 0))
            pred_im = pred_im[:, :, ::-1]  # convert BGR image to RGB image
            with PathManager.open(input["file_name"], "rb") as f:
                original_im = np.array(Image.open(f), dtype=np.float32)
            original_im = original_im / 255.
            rel_l1_err = np.abs(pred_im - original_im).sum() / original_im.sum()
            rel_l2_err = np.power(pred_im - original_im, 2).sum() / np.power(original_im, 2).sum()
            #
            self._rel_l1_err[input["image_id"]] = rel_l1_err * 100
            self._rel_l2_err[input["image_id"]] = rel_l2_err * 100

    def evaluate(self):
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "errors.txt")
            with PathManager.open(file_path, "w") as f:
                for image_id in self._rel_l1_err.keys():
                    f.write(f"{image_id} {self._rel_l1_err[image_id]:.2f} {self._rel_l2_err[image_id]:.2f}\n")

        mean_l1_err = sum(self._rel_l1_err.values()) / len(self._rel_l1_err)
        mean_l2_err = sum(self._rel_l2_err.values()) / len(self._rel_l2_err)

        res = {}
        res["l1_err"] = mean_l1_err
        res["l2_err"] = mean_l2_err
        results = OrderedDict({"inpainting": res})
        self._logger.info(results)
        return results
