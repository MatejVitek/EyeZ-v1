#!/usr/bin/env python3
from joblib import Parallel, delayed
import os
import re
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utils


def main():
	source = os.path.join(utils.get_rot_dir(), 'Recognition', 'all_directions_same_id', 'stage2')
	for (by, bins, bin_labels) in (('age', (25, 40), ('-25', '26-40', '41-')), ('gender', None, None)):
		target = os.path.join(source, '..', f'stage2_{by}')
		group_by(source, target, by, bins, bin_labels)


def group_by(source_dir, target_dir, by='age', bins=None, bin_labels=None):
	if not os.path.isdir(source_dir):
		raise ValueError(f"{source_dir} is not a directory.")

	dirs = [dir for dir in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, dir))]

	Parallel(n_jobs=-1, backend='multiprocessing')(
		delayed(shutil.copytree)(os.path.join(source_dir, dir), os.path.join(target_dir, bin.label, dir))
		for bin in utils.get_bins(dirs, by, bins, lambda dir: re.search(r'\d+', dir).group(), bin_labels)
		for dir in bin.samples
	)


if __name__ == '__main__':
	main()
