from detectron2.config import CfgNode as CN


def add_inpainter_config(cfg):
    _C = cfg

    # configuration for Inpainter
    _C.MODEL.INPAINTER = CN()
    _C.MODEL.INPAINTER.LOSS_REC_WEIGHT = 1.0
    _C.MODEL.INPAINTER.LOSS_GAN_WEIGHT = 1.0
    # configuration for Generator in Inpainter
    _C.MODEL.INPAINTER.GENERATOR = CN()
    _C.MODEL.INPAINTER.GENERATOR.NAME = ""
    _C.MODEL.INPAINTER.GENERATOR.CONV_DIMS = 24
    # configuration for Discriminator in Inpainter
    _C.MODEL.INPAINTER.DISCRIMINATOR = CN()
    _C.MODEL.INPAINTER.DISCRIMINATOR.NAME = ""
    _C.MODEL.INPAINTER.DISCRIMINATOR.CONV_DIMS = 64
    # configuration for random mask
    _C.INPUT.TRAIN_MASK_TYPE = ""
    _C.INPUT.MAX_SHIFT = 56
    # configuration for Adam
    _C.SOLVER.BETAS = (0.5, 0.999)
