# Deepfill v2 (PyTorch)

## Introduction
This repo is trying to reproduce some results of awesome 
Deepfill v2 ([Project](http://jiahuiyu.com/deepfill) & 
[Code](https://github.com/JiahuiYu/generative_inpainting)) 
because I personally prefer pytorh. Besides, I abuse the 
awesome [detectron2](https://github.com/facebookresearch/detectron2)
lib to implement although it is originally designed for 
object detection, because I appreciate its well-organized codes.
If you know more suitable tools, welcome to give recommendations.  

This repo is yet to be finished and tested.

## Progress
- [x] Build up the model.
- [x] Translate the pretrained tensorflow model into pytorch.
- [x] Fix the bug of converting tensorflow pretrained model to pytorch. 
Tensorflow behaves slightly different from Pytorch
on Conv2d when stride is greater than 1 (e.g. 2). Hence, I deal with
 this issue by manually striding the convolutional feature map.
 Moreover, original
tensorflow requires nearest neighbor downsample with 
`align_corners=True` while official pytorch `interpolate` does not
support `align_corners=True` when `mode="nearest"`. Therefore, 
I write my own downsampling functions enabling the `align_corners`. 
- [ ] Evaluate the pretrained model on Places2 and CelebA-HQ.
- [ ] Train the model on Places2 and CelebA-HQ.

## Run Demo
Now we can reproduce the demo results given in the original repo.
### prerequisites
* Python==3.6
* Pytorch==1.3.0 (**yet not tested for higher version**)
* detectron2==0.1

### pretrained model
The pretrained model is converted from tensorflow to pytorch using
`param_convertor.py`. You can download the tensorflow pretrained model 
[Places2](https://drive.google.com/drive/folders/1y7Irxm3HSHGvp546hZdAZwuNmhLUVcjO)
and convert the parameters or directly download the [converted model](
https://drive.google.com/file/d/1Q3p4Ejm1hm20cD2Hrk9beD-eG8Z-SM_O/view?usp=sharing
). Make sure the folder that contains the pretrained model is like
```
./output
./output/pretrained/
./output/pretrained/places2_256_deepfill_v2.pth 
```

### run demo
run the jupyter notebook file `./inpaint_demo.ipynb`. The results are
dumped in the folder `./demo_outputs`.

## Train model
TO BE COMPLETED

## Evaluate model
TO BE COMPLETED
