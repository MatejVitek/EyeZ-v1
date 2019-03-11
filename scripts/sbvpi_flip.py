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


def flip(source_dir, target_dir):
	if not os.path.isdir(source_dir):
		raise ValueError(f"{source_dir} is not a directory.")

	cls_dirs = [i for i in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, i))]
	max_cls = max(int(i) for i in cls_dirs)
	source_dirs = [os.path.join(source_dir, i) for i in cls_dirs]
	target_dirs_original = [os.path.join(target_dir, i) for i in cls_dirs]
	target_dirs_flipped = [os.path.join(target_dir, str(int(i) + max_cls)) for i in cls_dirs]
	for dir in target_dirs_original + target_dirs_flipped:
		os.makedirs(dir, exist_ok=True)
	
	Parallel(n_jobs=-1)(
		delayed(_process_image)(fname, source, target, flipped_target, max_cls)
		for (source, target, flipped_target) in zip(source_dirs, target_dirs_original, target_dirs_flipped)
		for fname in os.listdir(source)
	)


def _process_image(fname, source, target, flipped_target, max_cls):
	f = os.path.join(source, fname)
	basename, ext = os.path.splitext(fname)
	if not os.path.isfile(f) or ext.lower() not in IMG_EXTS or not re.match(r'\d+[LR]_[lrsu]_\d+', basename):
		return
		
	print(f"Processing file {fname}")
	
	target_f = os.path.join(target, fname)
	shutil.copyfile(f, target_f)

	img = Image.open(f).transpose(Image.FLIP_LEFT_RIGHT)
	new_basename = basename.split('_')
	new_basename[0] = str(int(new_basename[0][:-1]) + max_cls) + multi_replace(new_basename[0][-1], {'R': 'L', 'L': 'R'})
	new_basename[1] = multi_replace(new_basename[1], {'l': 'r', 'r': 'l'})
	new_basename = '_'.join(new_basename)
	img.save(os.path.join(flipped_target, f'{new_basename}{ext}'))


if __name__ == '__main__':
	source = sys.argv[1] if len(sys.argv) > 1 else os.path.join(get_eyez_dir(), 'SBVPI', 'SBVPI_with_masks')
	target = sys.argv[2] if len(sys.argv) > 2 else os.path.join(source, '..', 'SBVPI_mirrored')
	flip(source, target)

