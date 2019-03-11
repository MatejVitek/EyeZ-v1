#!/usr/bin/env python3
from keras import backend
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.generic_utils import CustomObjectScope
import numpy as np
import os
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import get_eyez_dir

sys.path.append(os.path.join(get_eyez_dir(), 'Recognition', 'Models', 'Lozej DeepLab+Xception'))
from deeplab import relu6, BilinearUpsampling


source = os.path.join('/home', 'matej', 'Downloads', 'MASD - Copy')
target = os.path.join('/home', 'matej', 'Downloads', 'MASD - Masks')
batch_size = 32
image_size = (320, 320)
input_channels = 3

with CustomObjectScope({'relu6': relu6, 'BilinearUpsampling': BilinearUpsampling}):
	model = load_model(os.path.join(get_eyez_dir(), 'Segmentation', 'Models', 'Sclera', 'DeepLab', 'dl_ssbc2019+sbvpi_unfrozen_021-0.863.hdf5'))

for root, _, files in os.walk(source):
	tgt_dir = os.path.join(target, os.path.relpath(root, source))
	os.makedirs(tgt_dir, exist_ok=True)
	for i in range(0, len(files), batch_size):
		size = min(batch_size, len(files) - i)
		batch = np.empty((size, *image_size, input_channels))
		for b, f in zip(range(size), range(i, i + size)):
			print(files[f])
			img = load_img(os.path.join(root, files[f]), target_size=image_size)
			batch[b] = img_to_array(img) / 255
		predictions = model.predict_on_batch(batch)
		for b, f in zip(range(size), range(i, i + size)):
			img = predictions[b].squeeze()
			img = (255 / img.max() * (img - img.min())).astype(np.uint8)
			Image.fromarray(img).save(os.path.join(tgt_dir, files[f]))

