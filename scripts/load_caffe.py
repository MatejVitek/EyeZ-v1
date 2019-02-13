#!/usr/bin/env python3
import caffe


caffe.set_mode_gpu()
model = caffe.Net('/media/matej/All Your Base/EyeZ/Rot/Segmentation/6.SegNet/6classes_4cv/1/Models/segnet_inference.prototxt', caffe.TEST, weights='/media/matej/All Your Base/EyeZ/Rot/Segmentation/6.SegNet/6classes_4cv/1/Models/snapshots/SSERBC_full_iter_75000.caffemodel')
