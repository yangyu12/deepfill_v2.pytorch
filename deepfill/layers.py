import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from detectron2.layers import Conv2d, cat

"""
This file contains the layers that are specific for GAN, including
- GatedConv2d
- GatedDeConv2d
- SpectralNormConv2d
- ContextAttention
"""


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class GatedConv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support zero-size tensor and more features.
    """
    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        # double the out_channels for gate
        self.num_channels = args[1]
        new_args = args[:1] + (2 * args[1], ) + args[2:]
        super().__init__(*new_args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        x, g = torch.split(x, self.num_channels, dim=1)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        x = x * g.sigmoid_()
        return x


class GatedDeConv2d(GatedConv2d):
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = super().forward(x)
        return x


class ContextAttention(torch.nn.Module):
    def __init__(self, kernel_size, stride, fuse_size, temperature=0.1):
        """
        :param kernel_size: the kernel size for matching, i.e. the size of extracted patch
        :param stride:
        :param fuse_size:
        :param temperature:
        """
        super().__init__()
        # TODO: do not support dilation currently
        # Note: the stride here is the rate argument in original tensorflow code.
        #  here dilation is assumed to be 1.
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.fuse_size = _pair(fuse_size)
        self.temperature = temperature

        # decide the deconv kernel size and downsampling scale
        self.deconv_kernel_size = tuple(2 * s for s in self.stride)
        self.downsampling_scale = tuple(1. / s for s in self.stride)

        # compute the padding for unfold, conv, and deconv aims at keeping the spatial size invariant
        self.unfold_deconv_padding = tuple((k - s) // 2 for k, s in zip(self.deconv_kernel_size, self.stride))
        self.unfold_conv_padding = tuple((k - 1) // 2 for k in self.kernel_size)
        self.conv_padding = self.unfold_conv_padding
        self.deconv_padding = self.unfold_deconv_padding

        # new a convolution kernel for fusing
        fk = fuse_size if isinstance(fuse_size, int) else fuse_size[0]
        self.register_buffer(
            "fuse_conv_weight",
            torch.eye(fk).view(1, 1, fk, fk)
        )
        self.fuse_conv_padding = _pair((fk - 1) // 2)

    def forward(self, fg_feature, bg_feature, masks):
        """

        :param fg_feature:
        :param bg_feature:
        :param mask:
        :return:
        """
        # 1. extract patches (2d x 2d) in background and reshape them as de-conv kernels
        deconv_weights = F.unfold(
            bg_feature, kernel_size=self.deconv_kernel_size, stride=self.stride, padding=self.unfold_deconv_padding
        )  # Tensor(N, C*2s*2s, (H/s)*(W/s))
        # out_h = (h + 2*ph - (k-1) - 1) / s + 1. let out_h = h/s --> ph = (k - s) / 2
        deconv_weights = self.reshape_to_batch_kernels(
            deconv_weights, bg_feature.size(1), self.deconv_kernel_size
        )  # Tensor(N, (H/s)*(W/s), C, 2s, 2s)

        # 2. downscaling foreground option: downscaling both foreground and background for matching
        # both are Tensor(N, C, H/s, W/s)
        fg_feature = F.interpolate(fg_feature, scale_factor=self.downsampling_scale)
        bg_feature = F.interpolate(bg_feature, scale_factor=self.downsampling_scale)
        # Note: the gradient problem may only exist for tf framework
        masks = F.interpolate(masks, scale_factor=self.downsampling_scale)  # Tensor(N, 1, H/s, W/s)

        # 3. extract patches (k x k) from downsampled background and reshape them as conv kernels
        conv_weights = F.unfold(
            bg_feature, kernel_size=self.kernel_size, padding=self.unfold_conv_padding
        )  # Tensor(N, C*k*k, (H/s)*(W/s))
        # out_h = h + 2*ph - (k-1). let out_h = h --> ph = (k - 1) / 2
        conv_weights = self.reshape_to_batch_kernels(
            conv_weights, bg_feature.size(1), self.kernel_size
        )  # Tensor(N, (H/s)*(W/s), C, k, k)

        # 4. process mask
        patch_masks = F.unfold(
            masks, kernel_size=self.kernel_size, padding=self.unfold_conv_padding
        )  # Tensor(N, 1*k*k, (H/s)*(W/s))
        patch_masks = (torch.mean(patch_masks, dim=1) == 0.)  # Tensor(N, (H/s)*(W/s))
        patch_masks = patch_masks[:, :, None, None]  # Tensor(N, (H/s)*(W/s), 1, 1)

        # 5. matching as convolution & reconstruction as deconvolution
        reconstructed_features = []
        matched_inds = []
        for fg_feat_per_im, conv_w_per_im, deconv_w_per_im, patch_mask_per_im in zip(
                fg_feature, conv_weights, deconv_weights, patch_masks
        ):
            # matching as convolution
            # normalize convolutional weights
            weight_norm = conv_w_per_im.flatten(start_dim=1).norm(dim=1, keepdim=True).clamp(min=1e-4)
            conv_w_per_im = conv_w_per_im / weight_norm[:, :, None, None]
            corr_response = F.conv2d(fg_feat_per_im[None], conv_w_per_im, padding=self.conv_padding)
            # Tensor(1, (H/s)*(W/s), H/s, W/s)

            # attention propagation
            corr_response = self.propagate_attention(corr_response)  # Tensor(1, (H/s)*(W/s), H/s, W/s)

            # softmax
            corr_response = torch.softmax(corr_response * patch_mask_per_im[None] / self.temperature, dim=1)
            corr_response = patch_mask_per_im[None] * corr_response  # Tensor(1, (H/s)*(W/s), H/s, W/s)

            # extract matching indexs
            matched_ind = torch.argmax(corr_response, dim=1)  # Tensor(1, H/s, W/s)
            map_width = fg_feat_per_im.size(2)  #
            matched_ind = torch.cat([matched_ind // map_width, matched_ind % map_width], dim=0)  # Tensor(2, H/s, W/s)

            # reconstruct as deconvolution
            rec_feature = F.conv_transpose2d(
                corr_response, deconv_w_per_im, stride=self.stride, padding=self.deconv_padding
            )  # out_h = (h / s - 1) * s - 2 * ph + (k - 1) + oph + 1. let out_h = h and oph = 0 --> ph = (k-s)/2 = s/2

            reconstructed_features.append(rec_feature / 4.)  # TODO: 4. seems to be arbitrary
            matched_inds.append(matched_ind)

        return cat(reconstructed_features, dim=0), torch.stack(matched_inds, dim=0)

    @classmethod
    def reshape_to_batch_kernels(cls, x, num_channels, kernel_size):
        # x: Tensor(N, C*kh*kw, L) --> Tensor(N, L, C, kh, kw)
        N, _, L = x.size()
        kh, kw = kernel_size
        return x.permute(0, 2, 1).reshape(N, L, num_channels, kh, kw)

    def propagate_attention(self, x):
        """

        :param x: Tensor(1, (H/s)*(W/s), H/s, W/s)
        :return:
        """
        _, L, H, W = x.size()
        assert L == H*W
        x = x.view(1, 1, L, -1)  # Tensor(1, 1, L, H/s * W/s)
        x = F.conv2d(x, self.fuse_conv_weight, padding=self.fuse_conv_padding)  # Tensor(1, 1, L, H/s * W/s)
        x = x.view(1, H, W, H, W)
        x = x.permute(0, 2, 1, 4, 3).contiguous().view(1, 1, L, -1)  # Tensor(1, (H/s) * (W/s), H/s * W/s)
        x = F.conv2d(x, self.fuse_conv_weight, padding=self.fuse_conv_padding)  # Tensor(1, 1, L, H/s * W/s)
        x = x.view(1, W, H, W, H)
        x = x.permute(0, 2, 1, 4, 3).contiguous().view(1, L, H, W)
        return x


# I think this is the correct way to build a spectral norm conv2d. The same way is used in
# https://github.com/knazeri/edge-connect/blob/ffdd3081db166d6954cc4e0254cfb04d24a2cb18/src/networks.py#L208
def SpectralNormConv2d(*args, **kwargs):
    return torch.nn.utils.spectral_norm(Conv2d(*args, **kwargs))
