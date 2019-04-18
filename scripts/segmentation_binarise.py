#!/usr/bin/env python3
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle
from sklearn.metrics import precision_recall_curve
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import get_eyez_dir


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
		
		
def binarise(source, target=None):
	if not os.path.isdir(source):
		raise ValueError(f"{source} is not a directory.")
	if not target:
		target = os.path.join(source, 'Binarised')
	
	Parallel(n_jobs=-1)(
		delayed(_process_file)(root, file, source, target)
		for root, _, files in os.walk(source)
		for file in files
	)


def _process_file(root, file, source, target):
	f = os.path.join(root, file)
	if '_gt' in f:
		print(f"{f} is a GT file")
		return
	if not os.path.isfile(f) or not os.path.splitext(f)[1].lower() in IMG_EXTS:
		print(f"{f} is not a valid prediction file")
		return
	gt_file = '_'.join(os.path.splitext(file)[0].split('_')[0:3] + ['gt.png'])
	gt_f = os.path.join(root, gt_file)
	if not os.path.isfile(gt_f):
		print(f"{f} missing corresponding GT file")
		return

	p = plt.imread(f)
	if p.ndim > 2:
		p = rgb2gray(p)
	m = plt.imread(gt_f)
	if m.ndim > 2:
		m = rgb2gray(m)
	# Binarise mask
	m = m.round().astype(int)
	if p.shape != m.shape:
		raise ValueError(f"Different dimensions ({p.shape} and {m.shape}) for base and GT mask in image {f}.")
	
	tgt_root = os.path.join(target, os.path.relpath(root, source))
	os.makedirs(tgt_root, exist_ok=True)
	threshold = find_max_f1(*precision_recall_curve(m.flatten(), p.flatten()))[0]
	print(f"{f}:\t{threshold}")
	plt.imsave(os.path.join(tgt_root, file), p >= threshold, cmap='gray')
	shutil.copyfile(gt_f, os.path.join(tgt_root, gt_file))


def f1_score(precision, recall):
	if precision == recall == 0:
		return 0
	return 2 * precision * recall / (precision + recall)

def find_max_f1(precision_v, recall_v, threshold_v):
	if len(threshold_v) < len(precision_v):
		threshold_v = np.append(threshold_v, [1.])
	max_f1 = (None, float('-inf'))
	for (p, r, t) in zip(precision_v, recall_v, threshold_v):
		f1 = f1_score(p, r)
		if f1 > max_f1[1]:
			max_f1 = (t, f1)
	return max_f1


def rgb2gray(rgba):
	return np.dot(rgba[...,:3], [0.2989, 0.587, 0.114])


if len(sys.argv) > 1:
	binarise(*sys.argv[1:])
else:
	binarise(os.path.join(get_eyez_dir(), 'Segmentation', 'Results', 'Vessels'))

