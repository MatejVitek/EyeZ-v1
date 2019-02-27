#!/usr/bin/env python3
from joblib import Parallel, delayed
import os
from PIL import Image
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utils


PRIMARY_CHANNELS = ('periocular', 'sclera')
SECONDARY_CHANNELS = ('canthus', 'eyelashes', 'iris', 'pupil', 'vessels')
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
NEW_SIZE = (480, 360)
NAMING = r'\d+[LR]_[lrsu]_\d+'


def main():
	source = os.path.join(utils.get_eyez_dir(), 'Recognition', 'Databases', 'Rot ScleraNet', 'stage2')
	target = os.path.join(utils.get_eyez_dir(), 'Resized_SegNet_Results')
	resize(source, target, NEW_SIZE, check_for_channels=False)


def resize(source_dir, target_dir, target_size, check_for_channels=True):
	if not os.path.isdir(source_dir):
		raise ValueError(f"{source_dir} is not a directory.")

	cls_dirs = [i for i in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, i))]
	source_dirs = [os.path.join(source_dir, cls_dir) for cls_dir in cls_dirs]
	target_dirs = [os.path.join(target_dir, cls_dir) for cls_dir in cls_dirs]
	for dir in target_dirs:
		os.makedirs(dir, exist_ok=True)
	
	Parallel(n_jobs=-1, backend='multiprocessing')(
		delayed(_resize_image)(fname, source, target, target_size, check_for_channels)
		for (source, target) in zip(source_dirs, target_dirs)
		for fname in os.listdir(source)
	)


def _resize_image(fname, source, target, target_size, check_for_channels):
	f = os.path.join(source, fname)
	basename, ext = os.path.splitext(fname)
	if not os.path.isfile(f) or ext.lower() not in IMG_EXTS or not re.fullmatch(NAMING, basename):
		return
	
	print(f"Processing file: {fname}")
	img = Image.open(f)
	img = img.resize(target_size, resample=Image.LANCZOS)
	img.save(os.path.join(target, fname))
	if not check_for_channels:
		return

	for channel in PRIMARY_CHANNELS + SECONDARY_CHANNELS:
		for ext in IMG_EXTS:
			fname = f'{basename}_{channel}{ext}'
			f = os.path.join(source, fname)
			if not os.path.isfile(f):
				continue

			print(f"Processing file: {fname}")
			img = Image.open(f)
			img = img.resize(target_size, resample=Image.NEAREST)
			img.save(os.path.join(target, fname))
			break

						
if __name__ == '__main__':
	main()
