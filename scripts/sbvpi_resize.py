#!/usr/bin/env python3
# Best structure 5 April 2019

from ast import literal_eval
from joblib import Parallel, delayed
import numpy as np
import os
from PIL import Image
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import get_eyez_dir


PRIMARY_CHANNELS = ('periocular', 'sclera')
SECONDARY_CHANNELS = ('canthus', 'eyelashes', 'iris', 'pupil', 'vessels')
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
NAMING = r'\d+[LR]_[lrsu]_\d+'


# Defaults
SIZE = (3000, 1700)
SRC = os.path.join(get_eyez_dir(), 'SBVPI', 'SBVPI_with_masks')


def resize(size=SIZE, source=SRC, target=None, check_for_channels=True, convert_original=False):
	if isinstance(size, str):
		size = literal_eval(size)
	if not os.path.isdir(source):
		raise ValueError(f"{source} is not a directory.")
	if not target:
		target = os.path.join(source, '..', 'Resized', 'x'.join(str(i) for i in size))
	if isinstance(check_for_channels, str):
		check_for_channels = literal_eval(check_for_channels)
	if isinstance(convert_original, str):
		convert_original = literal_eval(convert_original)
	
	Parallel(n_jobs=-1)(
		delayed(_process_file)(root, file, source, target, size, check_for_channels, convert_original)
		for root, _, files in os.walk(source)
		for file in files
	)


def _process_file(root, fname, source, target, size, check_for_channels, convert_original):
	f = os.path.join(root, fname)
	basename, ext = os.path.splitext(fname)
	if not os.path.isfile(f) or ext.lower() not in IMG_EXTS or not re.fullmatch(NAMING, basename):
		return
	
	print(f"Processing file: {fname}")
	tgt_root = os.path.join(target, os.path.relpath(root, source))
	os.makedirs(tgt_root, exist_ok=True)
	
	_resize(size, f, os.path.join(tgt_root, fname), convert=convert_original)
	if not check_for_channels:
		return

	for channel in PRIMARY_CHANNELS + SECONDARY_CHANNELS:
		for ext in IMG_EXTS:
			fname = f'{basename}_{channel}{ext}'
			f = os.path.join(root, fname)
			if not os.path.isfile(f):
				continue

			print(f"Processing file: {fname}")
			_resize(size, f, os.path.join(tgt_root, fname), convert=True)
			break
			
			
def _resize(size, src_f, tgt_f, convert=False):
	img = Image.open(src_f)
	resampler = Image.LANCZOS if all(new_size <= old_size for new_size, old_size in zip(size, img.size)) else Image.BICUBIC
	img = img.resize(size, resample=resampler)
	if convert:
		img = img.convert('L').point(lambda x: 255 if x >= 128 else 0, mode='1')
	img.save(tgt_f)


# Should include this check in case I ever want to import anything from this module
if __name__ == '__main__':
	resize(*sys.argv[1:])
