import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.layers import Conv2d, ShapeSpec

from deepfill.layers import GatedConv2d, GatedDeConv2d, ContextAttention, SpectralNormConv2d, align_corners_4x_nearest_downsample


@META_ARCH_REGISTRY.register()
class tf_GatedCNNSNPatchGAN(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.loss_rec_weight = cfg.MODEL.INPAINTER.LOSS_REC_WEIGHT
        self.loss_gan_weight = cfg.MODEL.INPAINTER.LOSS_GAN_WEIGHT

        # reuse exactly the same organization
        self.inpaint_net = GatedCNN(cfg, ShapeSpec(channels=5))

        self.normalizer = lambda x: x / 127.5 - 1.
        self.to(self.device)

    def forward(self, batched_inputs):
        # complete image
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.inpaint_net.size_divisibility)
        # triplet input maps:
        # erased regions
        masks = [x["mask"].to(self.device) for x in batched_inputs]
        masks = ImageList.from_tensors(masks, self.inpaint_net.size_divisibility)
        # mask the input image with masks
        erased_ims = images.tensor * (1. - masks.tensor)
        # ones map
        ones_ims = [torch.ones_like(x["mask"].to(self.device)) for x in batched_inputs]
        ones_ims = ImageList.from_tensors(ones_ims, self.inpaint_net.size_divisibility)
        # the conv layer use zero padding, this is used to indicate the image boundary

        # generation process
        input_tensor = torch.cat([erased_ims, ones_ims.tensor, masks.tensor], dim=1)
        coarse_inp, fine_inp, offset_flow = self.inpaint_net(input_tensor, masks.tensor)
        # offset_flow is used to visualize

        if self.training:
            raise NotImplementedError
        else:
            processed_results = []
            inpainted_im = erased_ims * (1. - masks.tensor) + fine_inp * masks.tensor
            for result, input_per_image, image_size in zip(inpainted_im, batched_inputs, images.image_sizes):
                height = input_per_image.get("height")
                width = input_per_image.get("width")
                r = sem_seg_postprocess(result, image_size, height, width)
                # abuse semantic segmentation postprocess. it basically does some resize
                processed_results.append({"inpainted": r})
            return processed_results

    def get_hidden_outputs(self, batched_inputs):
        # complete image
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.inpaint_net.size_divisibility)
        # triplet input maps:
        # erased regions
        masks = [x["mask"].to(self.device) for x in batched_inputs]
        masks = ImageList.from_tensors(masks, self.inpaint_net.size_divisibility)
        # mask the input image with masks
        erased_ims = images.tensor * (1. - masks.tensor)
        # ones map
        ones_ims = [torch.ones_like(x["mask"].to(self.device)) for x in batched_inputs]
        ones_ims = ImageList.from_tensors(ones_ims, self.inpaint_net.size_divisibility)
        # the conv layer use zero padding, this is used to indicate the image boundary

        # generation process
        input_tensor = torch.cat([erased_ims, ones_ims.tensor, masks.tensor], dim=1)

        all_hidden_outputs = self.inpaint_net.get_hidden_outputs(input_tensor, masks.tensor)
        # offset_flow is used to visualize

        return all_hidden_outputs



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
        #

        # stage 1: coarse network
        self.coarse_network = [
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
        ]
        convx_names = [
            "conv1",
            "conv2_downsample", "conv3",
            "conv4_downsample", "conv5", 'conv6',
            "conv7_atrous", "conv8_atrous", "conv9_atrous", "conv10_atrous",
            "conv11", "conv12",
            "conv13_upsample", "conv14",
            "conv15_upsample", "conv16", "conv17"
        ]
        for name, layer in zip(convx_names, self.coarse_network):
            self.add_module(name, layer)

        # stage 2: branch a: dilated gated conv
        self.refinement_conv_branch = [
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
        ]
        xconv_names = [
            "xconv1",
            "xconv2_downsample", "xconv3",
            "xconv4_downsample", "xconv5", "xconv6",
            "xconv7_atrous", "xconv8_atrous", "xconv9_atrous", "xconv10_atrous"
        ]
        for name, layer in zip(xconv_names, self.refinement_conv_branch):
            self.add_module(name, layer)

        # stage 2 branch b: with contextual attention module
        self.refinement_ctx_branch = [
            GatedConv2d(3, conv_dims, kernel_size=5, padding=2, activation=F.elu_),
            GatedConv2d(conv_dims, conv_dims, kernel_size=3, padding=1, stride=2, activation=F.elu_),  # /2x
            GatedConv2d(conv_dims, 2 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedConv2d(2 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, stride=2, activation=F.elu_),  # /4x
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.relu_),  # NOTE: the activation is relu
            # insert ctx module here
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_)
        ]
        self.ctx_module = ContextAttention(kernel_size=3, stride=2, fuse_size=3)  # ctx module
        pmconv_names = [
            "pmconv1",
            "pmconv2_downsample", "pmconv3",
            "pmconv4_downsample", "pmconv5", "pmconv6",
            "pmconv9", "pmconv10"
        ]
        for name, layer in zip(pmconv_names, self.refinement_ctx_branch):
            self.add_module(name, layer)

        # stage 2: refinement decoder
        self.refinement_decoder = [
            GatedConv2d(8 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedConv2d(4 * conv_dims, 4 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedDeConv2d(4 * conv_dims, 2 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),  # /2x
            GatedConv2d(2 * conv_dims, 2 * conv_dims, kernel_size=3, padding=1, activation=F.elu_),
            GatedDeConv2d(2 * conv_dims, conv_dims, kernel_size=3, padding=1, activation=F.elu_),  # /1x
            GatedConv2d(conv_dims, conv_dims // 2, kernel_size=3, padding=1, activation=F.elu_),
            Conv2d(conv_dims // 2, 3, kernel_size=3, padding=1, activation=F.tanh),  # the output layer is normal conv2d
        ]
        allconv_names = [
            "allconv11", "allconv12",
            "allconv13_upsample", "allconv14",
            "allconv15_upsample", "allconv16", "allconv17"
        ]
        for name, layer in zip(allconv_names, self.refinement_decoder):
            self.add_module(name, layer)


    @property
    def size_divisibility(self):
        return 8

    def forward(self, x, masks):
        xin = x
        # stage 1: coarse network
        for layer in self.coarse_network:
            x = layer(x)
        x_stage1 = x

        # stage2, paste result as input
        x = x * masks + xin[:, 0:3, :, :] * (1. - masks)
        # conv branch
        xnow = x
        for layer in self.refinement_conv_branch:
            x = layer(x)
        x_hallu = x
        # attention branch
        x = xnow
        for layer in self.refinement_ctx_branch[:6]:
            x = layer(x)
        # masks_s = F.interpolate(masks, scale_factor=1. / 2.)
        masks_s = align_corners_4x_nearest_downsample(masks)
        x, offset_flow = self.ctx_module(x, x, masks_s)
        for layer in self.refinement_ctx_branch[6:]:
            x = layer(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # decoder
        for layer in self.refinement_decoder:
            x = layer(x)
        x_stage2 = x

        return x_stage1, x_stage2, offset_flow

    def get_hidden_outputs(self, x, masks):
        all_hidden_outputs = []

        xin = x
        # stage 1: coarse network
        all_hidden_outputs.append(x)
        for layer in self.coarse_network:
            x = layer(x)
            all_hidden_outputs.append(x)
        x_stage1 = x

        # stage2, paste result as input
        x = x * masks + xin[:, 0:3, :, :] * (1. - masks)
        # conv branch
        xnow = x
        all_hidden_outputs.append(x)  #
        for layer in self.refinement_conv_branch:
            x = layer(x)
            all_hidden_outputs.append(x)  #
        x_hallu = x
        # attention branch
        x = xnow
        for layer in self.refinement_ctx_branch[:6]:
            x = layer(x)
            all_hidden_outputs.append(x)  #
        # masks_s = F.interpolate(masks, scale_factor=1. / 4.)
        masks_s = align_corners_4x_nearest_downsample(masks)
        x, offset_flow = self.ctx_module(x, x, masks_s)
        all_hidden_outputs.append(x)
        for layer in self.refinement_ctx_branch[6:]:
            x = layer(x)
            all_hidden_outputs.append(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # decoder
        all_hidden_outputs.append(x)  #
        for layer in self.refinement_decoder:
            x = layer(x)
            all_hidden_outputs.append(x)
        x_stage2 = x

        return all_hidden_outputs
