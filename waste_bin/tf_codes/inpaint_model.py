""" common model for DCGAN """
import logging

import cv2
import neuralgym as ng
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.layers import flatten, resize
from neuralgym.ops.gan_ops import gan_hinge_loss
from neuralgym.ops.gan_ops import random_interpolates

from .inpaint_ops import gen_conv, gen_deconv, dis_conv
from .inpaint_ops import random_bbox, bbox2mask, local_patch, brush_stroke_mask
from .inpaint_ops import resize_mask_like, contextual_attention


logger = logging.getLogger()


class InpaintCAModel(Model):
    def __init__(self):
        super().__init__('InpaintCAModel')

    def build_inpaint_net(self, x, mask, reuse=False,
                          training=True, padding='SAME', name='inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x, ones_x*mask], axis=3)

        all_hidden_outputs = []

        # two stage network
        cnum = 48
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gen_conv, gen_deconv],
                          training=training, padding=padding):
            # stage1
            all_hidden_outputs.append(x)  # include input
            x = gen_conv(x, cnum, 5, 1, name='conv1')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 2*cnum, 3, 2, name='conv2_downsample')  #
            all_hidden_outputs.append(x)
            x = gen_conv(x, 2*cnum, 3, 1, name='conv3')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, 2, name='conv4_downsample')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv5')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv6')
            all_hidden_outputs.append(x)
            mask_s = resize_mask_like(mask, x)
            x = gen_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv11')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv12')
            all_hidden_outputs.append(x)
            x = gen_deconv(x, 2*cnum, name='conv13_upsample')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 2*cnum, 3, 1, name='conv14')
            all_hidden_outputs.append(x)
            x = gen_deconv(x, cnum, name='conv15_upsample')
            all_hidden_outputs.append(x)
            x = gen_conv(x, cnum//2, 3, 1, name='conv16')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 3, 3, 1, activation=None, name='conv17')
            x = tf.nn.tanh(x)
            all_hidden_outputs.append(x)
            x_stage1 = x

            # stage2, paste result as input
            x = x*mask + xin[:, :, :, 0:3]*(1.-mask)
            x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())
            # conv branch
            # xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
            xnow = x
            all_hidden_outputs.append(x)
            x = gen_conv(xnow, cnum, 5, 1, name='xconv1')
            all_hidden_outputs.append(x)
            x = gen_conv(x, cnum, 3, 2, name='xconv2_downsample')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 2*cnum, 3, 1, name='xconv3')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 2*cnum, 3, 2, name='xconv4_downsample')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv5')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv6')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')
            all_hidden_outputs.append(x)
            x_hallu = x
            # attention branch
            x = gen_conv(xnow, cnum, 5, 1, name='pmconv1')
            all_hidden_outputs.append(x)
            x = gen_conv(x, cnum, 3, 2, name='pmconv2_downsample')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 2*cnum, 3, 1, name='pmconv3')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv5')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv6',
                                activation=tf.nn.relu)
            all_hidden_outputs.append(x)
            x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
            all_hidden_outputs.append(x)

            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv9')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv10')
            all_hidden_outputs.append(x)
            pm = x
            x = tf.concat([x_hallu, pm], axis=3)

            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, 1, name='allconv11')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 4*cnum, 3, 1, name='allconv12')
            all_hidden_outputs.append(x)
            x = gen_deconv(x, 2*cnum, name='allconv13_upsample')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 2*cnum, 3, 1, name='allconv14')
            all_hidden_outputs.append(x)
            x = gen_deconv(x, cnum, name='allconv15_upsample')
            all_hidden_outputs.append(x)
            x = gen_conv(x, cnum//2, 3, 1, name='allconv16')
            all_hidden_outputs.append(x)
            x = gen_conv(x, 3, 3, 1, activation=None, name='allconv17')
            x = tf.nn.tanh(x)
            all_hidden_outputs.append(x)
            x_stage2 = x
        return all_hidden_outputs

    def build_compare_graph(self, FLAGS, batch_data, reuse=False, is_training=False):
        """
        """
        # generate mask, 1 represents masked point
        if FLAGS.guided:
            batch_raw, edge, masks_raw = tf.split(batch_data, 3, axis=2)
            edge = edge[:, :, :, 0:1] / 255.
            edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)
        else:
            batch_raw, masks_raw = tf.split(batch_data, 2, axis=3)
        masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        batch_pos = batch_raw / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - masks)
        if FLAGS.guided:
            edge = edge * masks[:, :, :, 0:1]
            xin = tf.concat([batch_incomplete, edge], axis=3)
        else:
            xin = batch_incomplete
        # inpaint
        all_hidden_outputs = self.build_inpaint_net(
            xin, masks, reuse=reuse, training=is_training)

        return all_hidden_outputs
