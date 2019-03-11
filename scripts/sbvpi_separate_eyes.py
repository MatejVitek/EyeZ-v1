#!/usr/bin/env python3
from joblib import Parallel, delayed
import os
from PIL import Image
import re
import shutil
import sys

from mylibs.string import multi_replace

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import get_eyez_dir


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')


def separate_eyes(source_dir, target_dir):
	if not os.path.isdir(source_dir):
		raise ValueError(f"{source_dir} is not a directory.")

	cls_dirs = [i for i in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, i))]
	source_dirs = [os.path.join(source_dir, i) for i in cls_dirs]
	
	Parallel(n_jobs=-1)(
		delayed(_process_image)(fname, source, target_dir)
		for source in source_dirs
		for fname in os.listdir(source)
	)


def _process_image(fname, source, target_dir):
	f = os.path.join(source, fname)
	basename, ext = os.path.splitext(fname)
	if not os.path.isfile(f) or ext.lower() not in IMG_EXTS or not re.match(r'\d+[LR]_[lrsu]_\d+', basename):
		return
		
	print(f"Processing file {fname}")
	
	target = os.path.join(target_dir, basename.split('_')[0])
	os.makedirs(target, exist_ok=True)
	target_f = os.path.join(target, fname)
	shutil.copyfile(f, target_f)


if __name__ == '__main__':
	source = sys.argv[1] if len(sys.argv) > 1 else os.path.join(get_eyez_dir(), 'SBVPI', 'SBVPI_with_masks')
	target = sys.argv[2] if len(sys.argv) > 2 else os.path.join(source, '..', 'SBVP_separate_eyes')
	separate_eyes(source, target)

