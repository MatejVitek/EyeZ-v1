from joblib import Parallel, delayed
import logging
import matplotlib.image as img
import numpy as np
import os
import re
import shutil
import sys


PRIMARY_CHANNELS = ('periocular', 'sclera')
SECONDARY_CHANNELS = ('canthus', 'eyelashes', 'iris', 'pupil', 'vessels')
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')


def annotations_to_masks(source_dir, target_dir, save_type='json', **kwargs):
	if 'logging_file' not in kwargs:
		logging.basicConfig(
			stream=sys.stderr,
			level=kwargs.get('logging_level', logging.WARNING),
			format='[%(asctime)s] %(levelname)s: %(message)s',
			datefmt='%d-%m-%Y %H:%M%S'
		)
	else:
		logging.basicConfig(
			filename=kwargs['logging_file'] or f'{os.path.splitext(os.path.realpath(__file__))[0]}.log),
			filemode=kwargs.get('logging_filemode', 'w'),
			level=kwargs.get('logging_level', logging.WARNING),
			format='[%(asctime)s] %(levelname)s: %(message)s',
			datefmt='%d-%m-%Y %H:%M%S'
		)
	
	if isinstance(save_type, str):
		save_type = (save_type,)
	save_type = list(map(str.lower, save_type))
	all_types = any(t == 'all' for t in save_type)

	Parallel(n_jobs=-1, backend='multiprocessing')(
		delayed(_process_images)(i, source_dir, target_dir, save_type, all_types)
		for i in os.listdir(source_dir)
	)


def _process_images(i, source_dir, target_dir, save_type, all_types):
	source = os.path.join(source_dir, i)
	if not os.path.isdir(source):
		return
	target = os.path.join(target_dir, i)
	os.makedirs(target, exist_ok=True)

	for fname in os.listdir(source):
		f = os.path.join(source, fname)
		basename, ext = os.path.splitext(fname)
		if not os.path.isfile(f) or ext.lower() not in IMG_EXTS or not re.fullmatch(r'\d+[LR]_[lrsu]_\d+', basename):
			continue
		print(fname)
		shutil.copyfile(f, os.path.join(target, fname))

		mask = {}
		for channel in PRIMARY_CHANNELS + SECONDARY_CHANNELS:
			for img_ext in IMG_EXTS:
				try:
					annotation = img.imread(os.path.join(source, f'{basename}_{channel}{img_ext}'))[..., :3]
				except FileNotFoundError:
					continue
				except OSError:
					logging.error(f"Unexpected error while trying to read file {basename}{img_ext}. File may be corrupt.")
					continue
				mask[channel] = (annotation == (0, 1, 0)).all(axis=2)
				if channel == 'periocular':
					mask[channel] = ~mask[channel]
				break
			else:
				if channel in PRIMARY_CHANNELS:
					logging.warning(f"Missing primary channel {basename}_{channel}.")
				else:
					logging.info(f"Missing secondary channel {basename}_{channel}.")
		if not mask:
			continue

		save_name = os.path.join(target, '{}_{}{}')

		# Full image save
		if all_types or any(t in save_type for t in ('img', 'image', 'mask', 'channels')):
			for channel in mask:
				img.imsave(save_name.format(basename, channel, '.png'), mask[channel], cmap='gray')
		if any(f'{t}_full' in save_type for t in ('numpy', 'np', 'npy')):
			for channel in mask:
				np.save(save_name.format(basename, channel, '.npy'), mask[channel])

		# Sparse save
		for channel in mask.keys():
			mask[channel] = list(zip(*np.where(mask[channel])))
		if all_types or any(t in save_type or f'{t}_sparse' in save_type for t in ('numpy', 'np', 'npy')):
			for channel in mask:
				np.save(save_name.format(basename, channel, '.npy'), np.array(mask[channel]))

		# Sparse save as collection
		for channel in mask.keys():
			mask[channel] = [(int(x), int(y)) for (x, y) in mask[channel]]
		if all_types or 'json' in save_type:
			import json
			with open(save_name.format(basename, 'channels', '.json'), 'w') as save_file:
				json.dump(mask, save_file, indent=4, sort_keys=True)
		if all_types or 'csv' in save_type:
			import csv
			with open(save_name.format(basename, 'channels', '.csv'), 'w') as save_file:
				w = csv.DictWriter(save_file, mask.keys(), delimiter=';')
				w.writeheader()
				w.writerow(mask)
					
						
if __name__ == '__main__':
	source = os.path.join('/media', os.getlogin(), 'All Your Base', 'EyeZ', 'Rot', 'mag-peter-rot', 'SBVP_vessels')
	target = os.path.join(source, '..', 'SBVP_with_masks')
	annotations_to_masks(source, target, save_type=('numpy_full', 'img'), logging_level=logging.INFO, logging_file='')

