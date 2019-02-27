#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import scipy
import argparse
import math
import pylab
import datetime
from sklearn.preprocessing import normalize
# Change this to the absolute directoy to SegNet Caffe
caffe_root = '/hdd/EyeZ/Segmentation/Models/Rot SegNet/Code/caffe-segnet/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from timeit import default_timer as timer

caffe.set_mode_gpu()
model = caffe.Net('/hdd/EyeZ/Segmentation/Models/Rot SegNet/6classes_4cv/1/Models/segnet_inference.prototxt', caffe.TEST, weights='/hdd/EyeZ/Segmentation/Models/Rot SegNet/6classes_4cv/1/Models/snapshots/SSERBC_full_iter_75000.caffemodel')
