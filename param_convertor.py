"""
1. get the (name, value) of variables, where value is defaultly numpy arrays,
   and convert them to torch tensors.
"""
import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow
from collections import OrderedDict
import torch

latest_ckp = tf.train.latest_checkpoint('./output/pretrained/release_places2_256_deepfill_v2/')
reader = pywrap_tensorflow.NewCheckpointReader(latest_ckp)
name_params_map = OrderedDict()
vars_to_shape_map = reader.get_variable_to_shape_map()
for k in vars_to_shape_map.keys():
    name_params_map[k] = torch.from_numpy(np.asarray(reader.get_tensor(k)))
name_params_map = OrderedDict(sorted(name_params_map.items()))
for k, v in name_params_map.items():
    print(f"{k}:\t\t\t\t {v.size()}, {v.dtype}")

"""
2. map the (name, value) to the pytorch model. Basically, following are done
   a) change the name
   b) transpose, reshape, or slim the tensor
   c) only keep the generator parameters while remove the adam parameters, and discriminator parameters.
"""
# remove the optimization parameters
keys_to_delete = [k for k in name_params_map.keys() if "Adam" in k or "discriminator" in k]
keys_to_delete += ["lr", "beta2_power", "beta1_power", "global_step"]
for k in keys_to_delete:
    del name_params_map[k]
print("=============================================================")
print("after delte some parameters")
name_params_map = OrderedDict(sorted(name_params_map.items()))
for k, v in name_params_map.items():
    print(f"{k}:\t\t\t\t {v.size()}, {v.dtype}")

# config_file = "./configs/Inpainter/pretrained_eval.yaml"
# cfg = get_cfg()
# add_inpainter_config(cfg)
# cfg.merge_from_file(config_file)
# cfg.freeze()
# model = GatedCNNSNPatchGAN(cfg)
# for k, v in model.named_parameters():
#     print(f"{k}:\t\t\t\t{v.size()}")

def tf_name_to_torch_name(name):
    name = name.replace("inpaint_net", "generator")
    name = name.replace("/", ".")
    name = name.replace("kernel", "weight")
    middle_name = name.split(".")[1]
    stage_name = middle_name.split("_")[0]
    if stage_name.startswith("conv"):  # coarse network
        seq_id = int(stage_name[4:])
        new_middle_name = f"coarse_network.{seq_id-1}"
    elif stage_name.startswith("pmconv"):
        seq_id = int(stage_name[6:])
        if seq_id <= 6:
            new_middle_name = f"refinement_ctx_branch_1.{seq_id - 1}"
        else:
            new_middle_name = f"refinement_ctx_branch_2.{seq_id - 9}"
    elif stage_name.startswith("xconv"):
        seq_id = int(stage_name[5:])
        new_middle_name = f"refinement_conv_branch.{seq_id - 1}"
    elif stage_name.startswith("allconv"):
        seq_id = int(stage_name[7:])
        new_middle_name = f"refinement_decoder.{seq_id - 11}"
    else:
        raise KeyError(f"unexpected name: {name}")
    if "upsample" in middle_name:
        middle_name += f".{middle_name}_conv"
    name = name.replace(middle_name, new_middle_name)
    return name

name_params_map = {
    tf_name_to_torch_name(k): v.permute(3, 2, 0, 1).contiguous() if v.dim() == 4 else v
    for k, v in name_params_map.items()
}
name_params_map = OrderedDict(sorted(name_params_map.items()))
print("=============================================================")
print("after change name and change tensor shape")
for k, v in name_params_map.items():
    print(f"{k}:\t\t\t\t {v.size()}, {v.dtype}")

# save it as .pth file
with open("output/pretrained/places2_256_deepfill_v2.pth", "wb") as f:
    torch.save(name_params_map, f)
