import itertools
import numpy as np
import os

from naming_convention import NamingParser
import utils


L = 0
R = 1
C = 2
U = 3
EXTRA_INFO = utils.get_id_info()


class Sample(object):
	def __init__(self, f, naming):
		self.file = f
		self.basename = os.path.basename(os.path.splitext(self.file)[0])
		self.__dict__.update(naming.parse(self.basename))

		if EXTRA_INFO:
			self.gender = EXTRA_INFO[self.id].gender
			self.age = EXTRA_INFO[self.id].age
			self.color = EXTRA_INFO[self.id].color

		self.label = self.id


class Dataset(object):
	def __init__(self, dir, naming=None, **kw):
		"""
		Dataset handler

		:param dir: Root directory (should adhere to keras structure)
		:param naming: Naming convention handler. If None, will use default parser.
		:type  naming: NamingParser or None
		:param bool eyes_same_class: Whether both eyes should be counted as the same class.
		:param int mirrored_offset: Offset for mirrored identities. If 0, mirrored eyes will be counted as distinct classes.
		"""

		self.dir = dir
		self.naming = naming if naming else NamingParser()

		eyes_same_class = kw.get('both_eyes_same_class', True)
		mirrored_offset = kw.get('mirrored_offset', True)

		# Read data and determine class labels
		self.data = self._read_samples()
		flipped = [mirrored_offset and s.id > mirrored_offset for s in self.data]
		adjusted_ids = [(s.id - mirrored_offset) if f else s.id for (s, f) in zip(self.data, flipped)]
		sorted_ids = sorted(set(adjusted_ids))

		for (s, f, id) in zip(self.data, flipped, adjusted_ids):
			if eyes_same_class:
				s.label = sorted_ids.index(id)
			# If id is mirrored (and we're counting mirrored images as same class), we need to reverse L/R for mirrored images.
			else:
				s.label = 2 * sorted_ids.index(id) + ((1 - s.eye) if f else s.eye)

		self.n_classes = len({s.label for s in self.data})
		print(f"Found {len(self.data)} images belonging to {self.n_classes} classes.")

	def _read_samples(self):
		return [
			Sample(os.path.join(self.dir, cls_dir, sample), self.naming)
			for cls_dir in os.listdir(self.dir)
			if os.path.isdir(os.path.join(self.dir, cls_dir))
			for sample in os.listdir(os.path.join(self.dir, cls_dir))
			if os.path.isfile(os.path.join(self.dir, cls_dir, sample))
			and self.naming.valid(os.path.splitext(sample)[0])
		]

	def __iter__(self):
		return iter(self.data)


class GroupedDataset(Dataset):
	def __init__(self, dir, by, bins=None, bin_labels=None, **kw):
		"""
		Dataset split into groups by a specific sample attribute

		:param dir: See docs for :py:Dataset
		:param by: Key to group by. Can be a string (denoting the attribute name) or a function mapping a sample to a corresponding key value.
		:type  by: str or Callable
		:param bins: If [x1, ..., xn] is a sequence of numbers , will split data into n+1 bins (x<=x1, x1<x<=x2, ...).
		             Otherwise will split the data into n bins (x=x1, x=x2, ...).
		             If None, bins will be all values of :py:by over the dataset.
		:param bin_labels: Labels of bins. Should have size of n+1 or n, depending on the :py:bins parameter.
		                   If None, labels will be assigned automatically.
		:param kw: Other :py:Dataset keywords
		"""

		super().__init__(dir, **kw)

		f = (lambda s: getattr(s, by)) if isinstance(by, str) else by
		bins = sorted(bins or {f(s) for s in self.data})

		# Check if all bins are numbers
		try:
			all(bin <= float('inf') for bin in bins)

			# If bin labels not given, use "left_bin_border <= by < right_bin_border"
			if not bin_labels:
				bin_labels = [
					((str(bins[i - 1]) + " > ") if i > 0 else "")
					+ f'{by}'
					+ ((" <= " + str(bins[i])) if i < len(bins) else "")
					for i in range(len(bins) + 1)
				]
			self.groups = {
				label: Bin(label, [
					s
					for s in self.data
					if (i == 0 or f(s) > bins[i - 1])
					and (i == len(bins) or f(s) <= bins[i])
				])
				for i, label in enumerate(bin_labels)
			}

		# Else
		except TypeError:
			# If bin labels not given, use bin names
			if not bin_labels:
				bin_labels = bins
			self.groups = {label: Bin(label, [s for s in self.data if f(s) == bin]) for label, bin in zip(bin_labels, bins)}

	def __getitem__(self, group):
		return self.groups[group]

	def __contains__(self, item):
		if isinstance(item, Sample):
			return item in self.data
		else:
			return item in self.groups

	def all_groups(self):
		return self.groups.values()


class Bin(object):
	def __init__(self, label, samples):
		self.label = label
		self.samples = samples


class CVSplit(object):
	def __init__(self, data, n_folds):
		# Shuffle and split data list
		self.folds = np.array_split(utils.shuffle(data), n_folds)
		print(f"Splitting images into {n_folds} folds of size {len(self.folds[0])}.")

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
		print(f"Placing {self.split} images in gallery and {len(self.data) - self.split} in probe.")

	def __getitem__(self, index):
		if index == 0 or isinstance(index, str) and index.lower() in ('g', 'gallery'):
			return self.data[:self.split]
		elif index == 1 or isinstance(index, str) and index.lower() in ('p', 'probe'):
			return self.data[self.split:]
		else:
			raise IndexError("The only allowed indices are 0/'g'/'gallery' and 1/'p'/'probe'.")

	def shuffle(self):
		return utils.shuffle(self.data)
