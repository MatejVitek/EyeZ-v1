import itertools
import numpy as np
import os
import random

L = 0
R = 1
C = 2
U = 3

from naming import NamingParser
import utils


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

	def __eq__(self, other):
		if isinstance(other, Sample):
			return self.__dict__ == other.__dict__
		return NotImplemented

	def __hash__(self):
		return hash(tuple(sorted(self.__dict__.items())))


class Dataset(object):
	def __init__(self, dir=None, data=None, **kw):
		"""
		Dataset handler

		:param dir: Root directory (should adhere to keras structure)
		:param data: Collection of Samples to build the dataset from
		:param naming: Naming convention handler (dir mode only). If None, will use default parser.
		:type  naming: NamingParser or None
		:param bool both_eyes_same_class: Whether both eyes should be counted as the same class (dir mode only)
		:param int mirrored_offset: Offset for mirrored identities (dir mode only).
		                            If 0, mirrored eyes will be counted as distinct classes.
		"""

		if dir and data or not dir and not data:
			raise ValueError("Exactly one of dir/data should be passed to the Dataset constructor.")

		self.settings = kw

		if dir:
			self.data = self._read_samples(dir)
			self._assign_class_labels()
		else:
			self.data = data

		self.n_classes = len({s.label for s in self.data})

		if dir:
			print(f"Found {len(self.data)} images belonging to {self.n_classes} classes.")

	def _read_samples(self, dir):
		naming = self.settings.get('naming', NamingParser())
		return [
			Sample(os.path.join(dir, cls_dir, sample), naming)
			for cls_dir in os.listdir(dir)
			if os.path.isdir(os.path.join(dir, cls_dir))
			for sample in os.listdir(os.path.join(dir, cls_dir))
			if os.path.isfile(os.path.join(dir, cls_dir, sample))
			and naming.valid(os.path.splitext(sample)[0])
		]

	def _assign_class_labels(self):
		eyes_same_class = self.settings.get('both_eyes_same_class', True)
		mirrored_offset = self.settings.get('mirrored_offset', 0)

		flipped = [mirrored_offset and s.id > mirrored_offset for s in self.data]
		adjusted_ids = [(s.id - mirrored_offset) if f else s.id for (s, f) in zip(self.data, flipped)]
		sorted_ids = sorted(set(adjusted_ids))

		for (s, f, id) in zip(self.data, flipped, adjusted_ids):
			if eyes_same_class:
				s.label = sorted_ids.index(id)
			# If id is mirrored (and we're counting mirrored images as same class), we need to reverse L/R for mirrored images.
			else:
				s.label = 2 * sorted_ids.index(id) + ((1 - s.eye) if f else s.eye)

	def group_by(self, by, bins=None, bin_labels=None):
		"""
		Group Dataset by an attribute of the Samples

		:param by: Key to group by. Can be a string (denoting the attribute name) or a function mapping a sample to a corresponding key value.
		:type  by: str or Callable
		:param bins: If [x1, ..., xn] is a sequence of numbers , will split data into n+1 bins (x<=x1, x1<x<=x2, ...).
		             Otherwise will split the data into n bins (x=x1, x=x2, ...).
		             If None, bins will be all values of :py:by over the dataset.
		:param bin_labels: Labels of bins. Should have size of n+1 or n, depending on the :py:bins parameter.
		                   If None, labels will be assigned automatically.

		:return: Dictionary mapping labels to Datasets representing the groups.
		"""

		f = (lambda s: getattr(s, by)) if isinstance(by, str) else by
		bins = sorted(bins or {f(s) for s in self.data})

		# Check if all bins are numbers
		try:
			all(bin <= float('inf') for bin in bins)

			# If bin labels not given, use "left_bin_border < by <= right_bin_border"
			if not bin_labels:
				bin_labels = [
					((str(bins[i - 1]) + " < ") if i > 0 else "")
					+ f'{by}'
					+ ((" <= " + str(bins[i])) if i < len(bins) else "")
					for i in range(len(bins) + 1)
				]
			return {
				label: Dataset(data=[
					s
					for s in self.data
					if (i == 0 or f(s) > bins[i - 1])
					and (i == len(bins) or f(s) <= bins[i])
				], **self.settings)
				for i, label in enumerate(bin_labels)
			}

		# Else
		except TypeError:
			# If bin labels not given, use bin names
			if not bin_labels:
				bin_labels = bins
			return {
				label: Dataset(data=[s for s in self.data if f(s) == bin], **self.settings)
				for label, bin in zip(bin_labels, bins)
			}

	def shuffle(self):
		return utils.shuffle(self.data)

	def __getitem__(self, item):
		return self.data[item]

	def __iter__(self):
		return iter(self.data)

	def __len__(self):
		return len(self.data)

	def __add__(self, other):
		if isinstance(other, Dataset):
			return Dataset(data=list(set(self.data + other.data)), kw={**other.settings, **self.settings})
		return NotImplemented


class CVSplit(object):
	def __init__(self, dataset, n_folds):
		# Shuffle and split data list
		data_folds = np.array_split(dataset.shuffle(), n_folds)
		self.folds = [Dataset(data=fold, **dataset.settings) for fold in data_folds]
		print(f"Splitting images into {n_folds} folds of size {len(self.folds[0])}.")

	def __getitem__(self, index):
		try:
			# Return a single fold
			return self.folds[index]
		except TypeError:
			# Return multiple folds as a single Dataset
			return sum(self.folds[i] for i in index)

	def __len__(self):
		return len(self.folds)

	def __iter__(self):
		return iter(self.folds)


class GPSplit(object):
	def __init__(self, dataset):
		self.dataset = dataset
		self.gallery = None
		self.probe = None
		self.new_split()
		print(f"Placing {len(self.gallery)} images in gallery and {len(self.probe)} in probe.")

	def new_split(self):
		self.gallery = self._dataset(random.choices(self.dataset, k=len(self.dataset) // 2))
		self.probe = self._dataset(random.choices(self.dataset, k=len(self.dataset) // 2))
		return self.gallery, self.probe

	def _dataset(self, data):
		return Dataset(data=data, **self.dataset.settings)


class RatioSplit(GPSplit):
	def __init__(self, dataset, ratio):
		self.ratio = ratio
		super().__init__(dataset)

	def new_split(self, **kw):
		split = round(kw.get('ratio', self.ratio) * len(self.dataset))
		self.dataset.shuffle()
		self.gallery = self._dataset(self.dataset[:split])
		self.probe = self._dataset(self.dataset[split:])
		return self.gallery, self.probe


class AttributeSplit(GPSplit):
	def __init__(self, dataset, f, value=True):
		self.f = (lambda s: getattr(s, f)) if isinstance(f, str) else f
		self.value = value
		super().__init__(dataset)

	def new_split(self, **kw):
		f = kw.get('f', self.f)
		val = kw.get('value', self.value)
		self.gallery = self._dataset([s for s in self.dataset if f(s) == val])
		self.probe = self._dataset([s for s in self.dataset if f(s) != val])
		return self.gallery, self.probe


class BaseSplit(GPSplit):
	def new_split(self):
		self.dataset.shuffle()
		g = []
		p = []
		base_ns = {}

		for s in self.dataset:
			if s.label in base_ns:
				if s.n == base_ns[s.label]:
					g.append(s)
				else:
					p.append(s)
			else:
				base_ns[s.label] = s.n
				g.append(s)

		self.gallery = self._dataset(g)
		self.probe = self._dataset(p)
		return self.gallery, self.probe
