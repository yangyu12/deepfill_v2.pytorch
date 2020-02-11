import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.layers import Conv2d, ShapeSpec

from .layers import GatedConv2d, GatedDeConv2d, ContextAttention, SpectralNormConv2d


@META_ARCH_REGISTRY.register()
class GatedCNNSNPatchGAN(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.loss_rec_weight = cfg.MODEL.INPAINTER.LOSS_REC_WEIGHT
        self.loss_gan_weight = cfg.MODEL.INPAINTER.LOSS_GAN_WEIGHT

        gen_input_shape = ShapeSpec(channels=5)
        disc_input_shape = ShapeSpec(channels=4)
        self.generator = GatedCNN(cfg, gen_input_shape)
        self.discriminator = SNPatchDiscriminator(cfg, disc_input_shape)

        self.normalizer = lambda x: x / 127.5 - 1.
        self.to(self.device)

    def forward(self, batched_inputs):
        # complete image
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.generator.size_divisibility)
        # triplet input maps:
        # erased regions
        masks = [x["mask"].to(self.device) for x in batched_inputs]
        masks = ImageList.from_tensors(masks, self.generator.size_divisibility)
        # mask the input image with masks
        erased_ims = images.tensor * (1. - masks.tensor)
        # ones map
        ones_ims = [torch.ones_like(x["mask"].to(self.device)) for x in batched_inputs]
        ones_ims = ImageList.from_tensors(ones_ims, self.generator.size_divisibility)
        # the conv layer use zero padding, this is used to indicate the image boundary

        # generation process
        input_tensor = torch.cat([erased_ims, ones_ims.tensor, masks.tensor], dim=1)
        coarse_inp, fine_inp, offset_flow = self.generator(input_tensor, masks.tensor)
        # offset_flow is used to visualize

        if self.training:
            # reconstruction loss
            losses = {}
            losses["loss_coarse_rec"] = self.loss_rec_weight * torch.abs(images.tensor - coarse_inp).mean()
            losses["loss_fine_rec"] = self.loss_rec_weight * torch.abs(images.tensor - fine_inp).mean()

            # discriminator
            real_and_fake_ims = torch.cat([images.tensor, fine_inp], dim=0)
            real_and_fake_masks = torch.cat([masks.tensor, masks.tensor], dim=0)   # append masks
            disc_pred = self.discriminator(torch.cat([real_and_fake_ims, real_and_fake_masks], dim=1))
            pred_for_real, pred_for_fake = torch.split(disc_pred, disc_pred.size(0)//2, dim=0)
            # TODO: perhaps configure the loss function
            g_loss, d_loss = self.get_discriminator_hinge_loss(pred_for_real, pred_for_fake)
            losses['loss_gen'] = self.loss_gan_weight * g_loss
            losses['loss_disc'] = d_loss
            losses["generator_loss"] = sum([losses[k] for k in ["loss_coarse_rec", "loss_fine_rec", "loss_gen"]])
            losses["discriminator_loss"] = losses["loss_disc"]
            return losses
        else:
            processed_results = []
            inpainted_im = erased_ims * (1. - masks.tensor) + coarse_inp * masks.tensor
            for result, input_per_image, image_size in zip(inpainted_im, batched_inputs, images.image_sizes):
                height = input_per_image.get("height")
                width = input_per_image.get("width")
                r = sem_seg_postprocess(result, image_size, height, width)
                # abuse semantic segmentation postprocess. it basically does some resize
                processed_results.append({"inpainted": r})
            return processed_results

    @classmethod
    def get_discriminator_hinge_loss(cls, pred_r, pred_f):
        hinge_real = torch.mean(F.relu(1 - pred_r))
        hinge_fake = torch.mean(F.relu(1 + pred_f.detach()))  # discriminator loss is not used for updating generator
        d_loss = 0.5 * hinge_real + 0.5 * hinge_fake
        g_loss = -torch.mean(pred_f)
        return g_loss, d_loss


"""
generator
"""


class GatedCNN(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()

        #
        # it should be the half of the original configuration, because we latently double the dim inside the GatedConv2d
        conv_dims           = cfg.MODEL.INPAINTER.GENERATOR.CONV_DIMS
        in_channels         = input_shape.channels

        # TODO: This is ugly
        # stage 1: coarse network
        self.coarse_network = nn.Sequential(
            GatedConv2d(in_channels, conv_dims, kernel_size=5, padding=2, activation=F.elu_),
            GatedConv2d(conv_dims, 2 * conv_dims, kernel_size=3, padding=1, stride=2, activation=F.elu_),  # /2 x
            GatedConv2d(2 * conv_dims, 2 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedConv2d(2 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, stride=2, activation=F.elu_),  # /4 x
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),  # 5
            # Dialted Gated Conv Start
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=2, dilation=2, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=4, dilation=4, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=8, dilation=8, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=16, dilation=16, activation=F.elu_), # 9
            # Dialted Gated Conv End
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),  # 11
            GatedDeConv2d(4 * conv_dims, 2 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),  # /2 x
            GatedConv2d(2 * conv_dims, 2 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),  # 13
            GatedDeConv2d(2 * conv_dims, conv_dims, kernel_size=3, padding=1, activation=F.elu_),  # 1x
            GatedConv2d(conv_dims, conv_dims//2, kernel_size=3, padding=1, activation=F.elu_),
            Conv2d(conv_dims//2, 3, kernel_size=3, padding=1, activation=F.tanh),  # the output layer is normal conv2d
        )

        # stage 2 branch a: with contextual attention module
        self.refinement_ctx_branch_1 = nn.Sequential(
            GatedConv2d(3, conv_dims, kernel_size=5, padding=2, activation=F.elu_),
            GatedConv2d(conv_dims, conv_dims, kernel_size=3, padding=1, stride=2, activation=F.elu_),  # /2x
            GatedConv2d(conv_dims, 2 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedConv2d(2 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, stride=2, activation=F.elu_),  # /4x
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.relu_),  # NOTE: the activation is relu
        )
        self.ctx_module = ContextAttention(kernel_size=3, stride=2, fuse_size=3)  # ctx module
        self.refinement_ctx_branch_2 = nn.Sequential(
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
        )

        # stage 2: branch b: dilated gated conv
        self.refinement_conv_branch = nn.Sequential(
            GatedConv2d(3, conv_dims, kernel_size=5, padding=2, activation=F.elu_),
            GatedConv2d(conv_dims, conv_dims, kernel_size=3, padding=1, stride=2, activation=F.elu_),  # /2x
            GatedConv2d(conv_dims, 2 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedConv2d(2 * conv_dims, 2 * conv_dims, kernel_size=3, padding=1, stride=2, activation=F.elu_),  # /4x
            GatedConv2d(2 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),  # 5
            # Dialted Gated Conv Start
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=2, dilation=2, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=4, dilation=4, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=8, dilation=8, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=16, dilation=16, activation=F.elu_),
            # Dialted Gated Conv End
        )

        # stage 2: refinement decoder
        self.refinement_decoder = nn.Sequential(
            GatedConv2d(8 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedDeConv2d(4 * conv_dims, 2 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),  # /2x
            GatedConv2d(2 * conv_dims, 2 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedDeConv2d(2 * conv_dims, conv_dims, kernel_size=3, padding=1, activation=F.elu_),  # /1x
            GatedConv2d(conv_dims, conv_dims // 2, kernel_size=3, padding=1, activation=F.elu_),
            Conv2d(conv_dims // 2, 3, kernel_size=3, padding=1, activation=F.tanh),  # the output layer is normal conv2d
        )

    @property
    def size_divisibility(self):
        return 8

    def forward(self, x, masks):
        # stage 1: coarse network
        coarse_inp = self.coarse_network(x)

        # stage 2: refinement
        # branch a: contextual attention
        to_refine = coarse_inp * masks + x[:, :3, :, :] * (1. - masks)
        x = self.refinement_ctx_branch_1(to_refine)
        masks = F.interpolate(masks, scale_factor=1./4.)
        x, offset_flow = self.ctx_module(x, x, masks)
        ctx_out = self.refinement_ctx_branch_2(x)  # pm
        # branch b: gated conv
        conv_out = self.refinement_conv_branch(to_refine)  # x_hallu
        # decoder
        fine_inp = self.refinement_decoder(torch.cat([conv_out, ctx_out], dim=1))

        return coarse_inp, fine_inp, offset_flow


"""
discriminator
"""


class SNPatchDiscriminator(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()

        conv_dims               = cfg.MODEL.INPAINTER.DISCRIMINATOR.CONV_DIMS
        in_channels             = input_shape.channels

        #
        layers = []
        for out_channels in [
            conv_dims, 2 * conv_dims, 4 * conv_dims, 4 * conv_dims, 4 * conv_dims, 4 * conv_dims
        ]:  # /2x, /4x, /8x, /16x, /32x, /64x
            layers.append(
                SpectralNormConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=5,
                    padding=2,
                    stride=2,
                    activation=F.leaky_relu_
                )
            )
            in_channels = out_channels
        self.fully_conv_net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fully_conv_net(x)
        x = torch.flatten(x, start_dim=1)
        return x
