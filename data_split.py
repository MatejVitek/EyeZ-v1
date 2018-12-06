import itertools
import numpy as np

import utils


class CVSplit(object):
	def __init__(self, data, n_folds):
		# Shuffle and split data list
		self.folds = np.array_split(utils.shuffle(data), n_folds)
		print(f"Splitting training images into {n_folds} folds of size {len(self.folds[0])}.")

	def __getitem__(self, index):
		try:
			# Return a single fold
			return self.folds[index]
		except TypeError:
			# Return multiple folds as a single flattened list
			return list(itertools.chain.from_iterable(self.folds[i] for i in index))

	def __len__(self):
		return len(self.folds)


class RatioSplit(object):
	def __init__(self, data, ratio):
		self.data = data
		self.split = round(ratio * len(self.data))
		print(f"Placing {self.split} testing images in gallery and {len(self.data) - self.split} in probe.")

	def __getitem__(self, index):
		if index == 0 or isinstance(index, str) and index.lower() in ('g', 'gallery'):
			return self.data[:self.split]
		elif index == 1 or isinstance(index, str) and index.lower() in ('p', 'probe'):
			return self.data[self.split:]
		else:
			raise IndexError("The only allowed indices are 0/'g'/'gallery' and 1/'p'/'probe'.")

	def shuffle(self):
		return utils.shuffle(self.data)
