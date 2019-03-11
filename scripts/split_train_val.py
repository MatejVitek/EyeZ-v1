#!/usr/bin/env python3
from joblib import Parallel, delayed
import os
from random import shuffle
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import get_eyez_dir


def split(source_dir, train_dir, val_dir, source_gt_dir=None, train_gt_dir=None, val_gt_dir=None):
	if not os.path.isdir(source_dir):
		raise ValueError(f"{source_dir} is not a directory.")
	if source_gt_dir and not os.path.isdir(source_gt_dir):
		raise ValueError(f"{source_gt_dir} is not a directory.")
	if not all(dir is None for dir in (source_gt_dir, train_gt_dir, val_gt_dir)) and not all(dir is not None for dir in (source_gt_dir, train_gt_dir, val_gt_dir)):
		raise ValueError("Either pass all GT dirs or none of them.")

	sbvpi = []
	masd1 = []
	masd2 = []

	for image in os.listdir(source_dir):
		id = int(image.split('_')[0][:-1])
		if id < 100:
			sbvpi.append(image)
		elif id < 200:
			masd1.append(image)
		else:
			masd2.append(image)

	for dir in (train_dir, val_dir, train_gt_dir, val_gt_dir):
		if dir:
			os.makedirs(dir, exist_ok=True)
	
	with Parallel(n_jobs=-1) as parallel:
		for data in (sbvpi, masd1, masd2):
			shuffle(data)
			split = round(0.8 * len(data))
			parallel(
				delayed(_process_image)(image, (source_dir, source_gt_dir), (train_dir, train_gt_dir) if i < split else (val_dir, val_gt_dir))
				for i, image in enumerate(data)
			)
	
def _process_image(image, src, tgt):
	print(image)
	source_dir, source_gt_dir = src
	target_dir, target_gt_dir = tgt
	
	shutil.copy(os.path.join(source_dir, image), target_dir)
	if source_gt_dir and target_gt_dir:
		for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.eps'):
			try:
				shutil.copy(os.path.join(source_gt_dir, f'{os.path.splitext(image)[0]}{ext}'), target_gt_dir)
				break
			except FileNotFoundError:
				pass


if __name__ == '__main__':
	if len(sys.argv) > 1:
		split(*sys.argv)
	else:
		args = [None] * 6
		args[0] = os.path.join(get_eyez_dir(), 'Segmentation', 'Databases', 'Sclera', 'SSBC2019 + SBVPI', 'Images')
		args[1] = os.path.join(args[0], '..', 'train', 'Images')
		args[2] = os.path.join(args[0], '..', 'val', 'Images')
		args[3] = os.path.join(args[0], '..', 'Masks')
		args[4] = os.path.join(args[0], '..','train', 'Masks')
		args[5] = os.path.join(args[0], '..','val', 'Masks')
		split(*args)

