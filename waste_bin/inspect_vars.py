import os
from pprint import pprint
import tensorflow as tf
from collections import OrderedDict
import torch

tf_path = os.path.abspath('./output/pretrained/release_places2_256_deepfill_v2/snap-0')  # Path to our TensorFlow checkpoint
init_vars = tf.train.list_variables(tf_path)
tf_vars = []
tf_vars_types = []
for name, shape in init_vars:
    print("Loading TF weight {} with shape {}".format(name, shape))
    array = tf.train.load_variable(tf_path, name)
    tf_vars.append((name, array.squeeze()))
    tf_vars_types.append((name, array.dtype))

# convert to pytorch
def tf_name_to_torch_name(name):
    name = name.replace("/", ".")
    name = name.replace("kernel", "weight")
    if "conv13_upsample" in name:
        name = name.replace("conv13_upsample.conv13_upsample_conv", "conv13_upsample")
    if "conv15_upsample" in name:
        name = name.replace("conv15_upsample.conv15_upsample_conv", "conv15_upsample")
    if "allconv13_upsample" in name:
        name = name.replace("allconv13_upsample.allconv13_upsample_conv", "allconv13_upsample")
    if "allconv15_upsample" in name:
        name = name.replace("allconv15_upsample.allconv15_upsample_conv", "allconv15_upsample")
    return name

keys_to_delete = [x[0] for x in tf_vars if "Adam" in x[0] or "discriminator" in x[0]]
keys_to_delete += ["lr", "beta2_power", "beta1_power", "global_step"]
remained_tf_vars = [x for x in tf_vars if x[0] not in keys_to_delete]
tf_vars_dict = [(tf_name_to_torch_name(x[0]), torch.from_numpy(x[1])) for x in remained_tf_vars]
tf_vars_dict = {
    k: v.permute(3, 2, 0, 1).contiguous() if v.dim() == 4 else v for k, v in tf_vars_dict
}
tf_vars_dict = OrderedDict(sorted(tf_vars_dict.items()))
pprint(tf_vars_dict.keys())

# save it as .pth file
with open("output/pretrained/tf_places2_256_deepfill_v2.pth", "wb") as f:
    torch.save(tf_vars_dict, f)
