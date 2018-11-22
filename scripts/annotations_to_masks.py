from collections import Counter
from joblib import Parallel, delayed
import logging
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import os
import platform
import re
import shutil
import sys


PATH = 'D:' if platform.system().lower() == 'windows' else os.path.join('/media', os.getlogin(), 'All Your Base')
PRIMARY_CHANNELS = ('periocular', 'sclera')
SECONDARY_CHANNELS = ('canthus', 'eyelashes', 'iris', 'pupil', 'vessels')
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')


def annotations_to_masks(source_dir, target_dir, save_type='json', plot=False, overwrite=True, **kwargs):
	if 'logging_file' not in kwargs:
		logging.basicConfig(
			stream=sys.stderr,
			level=kwargs.get('logging_level', logging.WARNING),
			format='[%(asctime)s] %(levelname)s: %(message)s',
			datefmt='%d-%m-%Y %H:%M%S'
		)
	else:
		logging.basicConfig(
			filename=kwargs['logging_file'] or f'{os.path.splitext(os.path.realpath(__file__))[0]}.log',
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
		delayed(_process_images)(i, source_dir, target_dir, overwrite, save_type, all_types, plot)
		for i in '1'#os.listdir(source_dir)
	)


def _process_images(i, source_dir, target_dir, overwrite, save_type, all_types, plot):
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
		print(f"Processing file {fname}")
		shutil.copyfile(f, os.path.join(target, fname))

		mask = {}
		n_masks = Counter()
		channels = PRIMARY_CHANNELS + SECONDARY_CHANNELS
		for channel in channels:
			for img_ext in IMG_EXTS:
				try_file = f'{basename}_{channel}{img_ext}'
				try:
					annotation = img.imread(os.path.join(source, try_file))[..., :3]
				except FileNotFoundError:
					continue
				except OSError:
					logging.error(f"Unexpected error while trying to read file {try_file}. File may be corrupt.")
					continue
					
				mask[channel] = np.isclose(annotation, (0, 1, 0), atol=.01, rtol=0).all(axis=2)
				if channel == 'periocular':
					mask[channel] = ~mask[channel]
				break
				
				n_masks[channel] += 1
				if plot:
					fig = plt.figure(num=f'{try_file}')
					fig.add_subplot(121)
					plt.imshow(annotation)
					fig.add_subplot(122)
					plt.imshow(mask[channel])
					plt.show()
			else:
				if channel in PRIMARY_CHANNELS:
					logging.warning(f"Missing primary channel {try_file}.")
				else:
					logging.info(f"Missing secondary channel {try_file}.")
		if not mask:
			continue

		save_name = os.path.join(target, '{}_{}{}')

		# Full image save
		if all_types or any(t in save_type for t in ('img', 'image', 'mask')):
			for channel in mask:
				name = save_name.format(basename, channel, '.png')
				if overwrite or not os.path.exists(name):
					img.imsave(save_name.format(basename, channel, '.png'), mask[channel], cmap='gray')
		if any(f'{t}_full' in save_type for t in ('numpy', 'np', 'npy')):
			for channel in mask:
				name = save_name.format(basename, channel, '.npy')
				if overwrite or not os.path.exists(name):
					np.save(name, mask[channel])

		# Sparse save
		for channel in mask.keys():
			mask[channel] = list(zip(*np.where(mask[channel])))
		if all_types or any(t in save_type or f'{t}_sparse' in save_type for t in ('numpy', 'np', 'npy')):
			for channel in mask:
				name = save_name.format(basename, channel, '.npy')
				if overwrite or not os.path.exists(name):
					np.save(save_name.format(basename, channel, '.npy'), np.array(mask[channel]))

		# Sparse save as collection
		for channel in mask.keys():
			mask[channel] = [(int(x), int(y)) for (x, y) in mask[channel]]
		if all_types or 'json' in save_type:
			import json
			try:
				with open(save_name.format(basename, 'channels', '.json'), 'w' if overwrite else 'x') as save_file:
					json.dump(mask, save_file, indent=4, sort_keys=True)
			except FileExistsError:
				pass
		if all_types or 'csv' in save_type:
			import csv
			try:
				with open(save_name.format(basename, 'channels', '.csv'), 'w' if overwrite else 'x') as save_file:
					w = csv.DictWriter(save_file, mask.keys(), delimiter=';')
					w.writeheader()
					w.writerow(mask)
			except FileExistsError:
				pass
			
	max_len = max(map(len, channels)), max(map(len, map(str, missing_masks.values())))
	print("Number of found annotations:")
	print(*(f"{channel:{max_len[0]}}: {missing_masks[channel]:{max_len[1]}}" for channel in channels), sep="\n")
					
						
if __name__ == '__main__':
	source = os.path.join(PATH, 'EyeZ', 'Rot', 'mag-peter-rot', 'SBVP_vessels')
	target = os.path.join(source, '..', 'SBVP_with_masks')
	annotations_to_masks(source, target, save_type='img', plot=False, overwrite=True, logging_level=logging.WARNING, logging_file='')
