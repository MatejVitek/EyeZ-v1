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
import subprocess
import sys


PATH = 'D:' if platform.system().lower() == 'windows' else os.path.join('/media', os.getlogin(), 'All Your Base')
PRIMARY_CHANNELS = ('periocular', 'sclera')
SECONDARY_CHANNELS = ('canthus', 'eyelashes', 'iris', 'pupil', 'vessels')
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')


def annotations_to_masks(source_dir, target_dir, **kwargs):
	if not os.path.isdir(source_dir):
		raise ValueError(f"{source_dir} is not a directory.")
		
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

	plot = kwargs.get('plot', False)
	overwrite = kwargs.get('overwrite', True)
	count_only = kwargs.get('count_only', False)
	log_diff = kwargs.get('log_diff', True)
	
	save_type = kwargs.get('save_type', 'img')
	if save_type:
		if isinstance(save_type, str):
			save_type = (save_type,)
		save_type = list(map(str.lower, save_type))
		all_types = any(t == 'all' for t in save_type)
	else:
		all_types = None
	
	cls_dirs = [i for i in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, i))]
	source_dirs = [os.path.join(source_dir, cls_dir) for cls_dir in cls_dirs]
	target_dirs = [os.path.join(target_dir, cls_dir) for cls_dir in cls_dirs]
	for dir in target_dirs:
		os.makedirs(dir, exist_ok=True)
	channels = PRIMARY_CHANNELS + SECONDARY_CHANNELS
	
	counter = sum(Parallel(n_jobs=-1, backend='multiprocessing')(
		delayed(_process_image)(fname, source, target, channels, overwrite, save_type, all_types, plot)
		for (source, target) in zip(source_dirs, target_dirs)
		for fname in os.listdir(source)
	), Counter({'class': len(cls_dirs)}))

	max_len = max(map(len, channels)), max(map(len, map(str, counter.values())))
	log = f"Classes: {counter['class']}\nImages: {counter['image']}\nAnnotations: {counter['channel']}\n"
	log += "\n".join(f"{channel:{max_len[0]}}: {counter[channel]:{max_len[1]}}" for channel in channels)
	
	if log_diff:
		with open('./tree_src', 'w') as f:
			subprocess.run(['tree', source_dir], stdout=f)
		with open('./tree_tgt', 'w') as f:
			subprocess.run(['tree', target_dir], stdout=f)
		log += "\n\n\n" + subprocess.run(['diff', './tree_src', './tree_tgt'], stdout=subprocess.PIPE, universal_newlines=True).stdout
		os.remove('./tree_src')
		os.remove('./tree_tgt')
	
	print(log)
	logging.log(1000, log)


def _process_image(fname, source, target, channels, overwrite, save_type, all_types, plot):
	counter = Counter()
	
	f = os.path.join(source, fname)
	basename, ext = os.path.splitext(fname)
	if not os.path.isfile(f) or ext.lower() not in IMG_EXTS or not re.fullmatch(r'\d+[LR]_[lrsu]_\d+', basename):
		return counter
		
	print(f"Processing file {fname}")
	counter['image'] += 1
	
	target_f = os.path.join(target, fname)
	if overwrite or not os.path.exists(target_f):
		shutil.copyfile(f, target_f)

	mask = {}
	for channel in channels:
		for img_ext in IMG_EXTS:
			try_fname = f'{basename}_{channel}{img_ext}'
			try_f = os.path.join(source, try_fname)
			if not save_type:
				if os.path.isfile(try_f):
					counter['channel'] += 1
					counter[channel] += 1
					break
				else:
					continue
			try:
				annotation = img.imread(try_f)[..., :3]
			except FileNotFoundError:
				continue
			except OSError:
				logging.error(f"Unexpected error while trying to read file {try_fname}. File may be corrupt.")
				continue
				
			mask[channel] = np.isclose(annotation, (0, 1, 0), atol=.01, rtol=0).all(axis=2)
			if channel == 'periocular':
				mask[channel] = ~mask[channel]
			counter['channel'] += 1
			counter[channel] += 1
			
			if plot:
				fig = plt.figure(num=f'{try_fname}')
				fig.add_subplot(121)
				plt.imshow(annotation)
				fig.add_subplot(122)
				plt.imshow(mask[channel])
				plt.show()
				
			break
		else:
			if channel in PRIMARY_CHANNELS:
				logging.warning(f"Missing primary channel {basename}_{channel}.")
			else:
				logging.info(f"Missing secondary channel {basename}_{channel}.")
	if not mask or not save_type:
		return counter

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
	
	return counter
					
						
if __name__ == '__main__':
	source = os.path.join(PATH, 'EyeZ', 'Rot', 'SBVP_vessels')
	target = os.path.join(source, '..', 'SBVP_with_masks')
	annotations_to_masks(source, target, save_type='img', plot=False, overwrite=False, logging_level=logging.WARNING, logging_file='')
