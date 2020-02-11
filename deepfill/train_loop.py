import os
import logging
import time
from typing import Any, Dict, List
import torch

from detectron2.engine import DefaultTrainer, hooks
from detectron2.data import build_detection_train_loader, build_detection_test_loader


from .dataset import InpainterDatasetMapper
from .inpainting_evaluation import InpaintingEvaluator


class InpainterTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        g_optm, d_optm = self.build_gen_disc_optimizer(cfg)
        self.generator_optimizer = g_optm
        self.discriminator_optimizer = d_optm

        # delete LRSchedule hook
        self._hooks = [h for h in self._hooks if not isinstance(h, hooks.LRScheduler)]

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=InpainterDatasetMapper(cfg, True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=InpainterDatasetMapper(cfg, False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return InpaintingEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

    def only_load_model(self):
        checkpoint = self.checkpointer._load_file(self.cfg.MODEL.WEIGHTS)
        self.checkpointer._load_model(checkpoint)
        # TODO: method that automatically remap weight names should be written
        logger = logging.getLogger(__name__)
        logger.info("Loaded model from {}".format(self.cfg.MODEL.WEIGHTS))

    @classmethod
    def build_optimizer(cls, cfg, model):
        return None

    def build_gen_disc_optimizer(self, cfg):
        # build optimizer for generator
        gen_optimizer = torch.optim.Adam(
            params=self.model.generator.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS
        )
        # build optimizer for discriminator
        disc_optimizer = torch.optim.Adam(
            params=self.model.discriminator.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS
        )
        return gen_optimizer, disc_optimizer

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If your want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        losses = sum(loss for loss in loss_dict.values())  # this losses is only used for anormal detection
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        One step for generator
        """
        self.generator_optimizer.zero_grad()
        loss_dict["generator_loss"].backward()
        self.generator_optimizer.step()

        """
        One step for discriminator
        """
        self.discriminator_optimizer.zero_grad()
        loss_dict["discriminator_loss"].backward()
        self.discriminator_optimizer.step()
