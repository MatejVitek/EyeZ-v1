L = 0
R = 1
C = 2
U = 3


import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp

from keras.layers import Dense, Flatten
from keras.models import Sequential

from data_split import CVSplit, RatioSplit
from image_generators import ImageGenerator, LabeledImageGenerator
from model_wrapper import CVModel, TrainableNNModel
from naming_convention import NamingHandler
import utils


EXTRA_INFO = utils.get_id_info()


class Sample(object):
	def __init__(self, f, info):
		self.file = f
		self.basename = os.path.basename(os.path.splitext(self.file)[0])
		self.__dict__.update(info.naming.parse(self.basename))

		self.gender = EXTRA_INFO[self.id].gender
		self.age = EXTRA_INFO[self.id].age
		self.color = EXTRA_INFO[self.id].color

		self.label = self.id


class CV(object):
	def __init__(self, model, train_dir, test_dir, **kw):
		"""
		Initializes a new cross-validation object

		:param CVModel model: Predictor wrapped for evaluation
		:param train_dir: Directory with training data (must adhere to keras structure). If None, no training will be done.
		:type  train_dir: str or None
		:param str test_dir: Directory with testing data (must adhere to keras structure)
		:param bool eyes_same_class: Whether both eyes should be counted as the same class.
		:param int mirrored_offset: Offset for mirrored identities. If 0, mirrored eyes will be counted as distinct classes.
		:param bool plot: Whether to draw plots
		:param group_by: Characteristic to group identities by. If None, no grouping will be done.
		:param bins: If [x1, ..., xn] is a sequence of numbers , will split data into n+1 bins (x<=x1, x1<x<=x2, ...).
		             Otherwise will split the data into n bins (x=x1, x=x2, ...).
		:param bool crossbin_eval: Whether to use cross-bin evaluation in impostor testing
		"""

		# Model and directories
		self.model = model
		self.train_dir = train_dir
		self.test_dir = test_dir

		# If training directory was specified, model has to be trainable (uncomment after CVModel done)
		if self.train_dir and not isinstance(self.model, TrainableNNModel):
			raise ValueError("If training, model must be a subclass of TrainableNNModel")
		
		# Other settings
		eyes_same_class = kw.get('both_eyes_same_class', True)
		mirrored_offset = kw.get('mirrored_offset', True)
		self.plot = kw.get('plot', True)
		self.grp_by = kw.get('group_by')
		self.grp_bins = kw.get('bins')
		self.grp_cross_eval = kw.get('crossbin_eval')

		# Uninitialized variables
		self.tv_split = self.gp_split = None
		self.color = None
		self.bin_labels = None
		
		# File naming conventions
		self.naming = NamingHandler(
			kw.get('naming_re'),
			kw.get('naming', r'ie_d_n'),
			kw.get('eyes', r'LR'),
			kw.get('directions', r'lrsu'),
			kw.get('naming_strict', False)
		)

		# Read data and determine class labels
		self.data = {}
		self.n_classes = {}
		for (t, dir) in (('train', self.train_dir), ('test', self.test_dir)):
			if not dir:
				continue

			self.data[t] = self._read_samples(dir)
			flipped = [mirrored_offset and s.id > mirrored_offset for s in self.data[t]]
			adjusted_ids = [(s.id - mirrored_offset) if f else s.id for (s, f) in zip(self.data[t], flipped)]
			sorted_ids = sorted({_ for _ in adjusted_ids})

			for (s, f, id) in zip(self.data[t], flipped, adjusted_ids):
				if eyes_same_class:
					s.label = sorted_ids.index(id)
				# If id is mirrored (and we're counting mirrored images as same class), we need to reverse L/R for mirrored images.
				else:
					s.label = 2 * sorted_ids.index(id) + ((1 - s.eye) if f else s.eye)

			self.n_classes[t] = len({s.label for s in self.data[t]})
			print(f"Found {len(self.data[t])} {t}ing images belonging to {self.n_classes[t]} classes.")

		print(f"Found {sum(len(data) for data in self.data.values())} total images belonging to {len({s.label for data in self.data.values() for s in data})} classes.")

	def __call__(self, *args, **kw):
		return self.cross_validate(*args, **kw)

	def cross_validate(self, k=10, gp_split=0.3):
		# Uniform random colors
		if self.plot:
			plt.ion()
			colors = np.ones((k, 3))
			colors[:, 0] = np.linspace(0, 1, k, endpoint=False)
			colors = matplotlib.colors.hsv_to_rgb(colors)

		# Special case for k = 1
		run_once = False
		if k <= 1:
			k = 2
			run_once = True

		# Group data (only one group if no grouping done)
		self.data = {t: [data] for (t, data) in self.data.items()}

		if self.train_dir:
			if self.grp_by:
				self.data['train'] = self._group_samples(self.data['train'][0])
		else:
			self.data['train'] = [None]

		if self.grp_by:
			self.data['test'] = self._group_samples(self.data['test'][0])

		if len(self.data['test']) == 1 < len(self.data['train']):
			self.data['test'] *= len(self.data['train'])
		elif len(self.data['train']) == 1 < len(self.data['test']):
			self.data['train'] *= len(self.data['test'])
		if not self.bin_labels:
			self.bin_labels = [None] * len(self.data['train'])
		assert len(self.data['train']) == len(self.data['test']) == len(self.bin_labels)

		evaluation = None
		for train_group, test_group, bin in zip(self.data['train'], self.data['test'], self.bin_labels):
			if self.grp_by:
				print(bin.format(self.grp_by if isinstance(self.grp_by, str) else "x") + ":")

			if self.train_dir:
				self.tv_split = CVSplit(train_group, k)
			self.gp_split = RatioSplit(test_group, gp_split)

			for fold in range(k):
				print(f"Fold {fold+1}:")

				if self.plot:
					self.color = colors[fold]

				if self.train_dir:
					assert isinstance(self.model, TrainableNNModel)
					train = self.tv_split[(x for x in range(len(self.tv_split)) if x != fold)]
					val = self.tv_split[fold]
					self.model.train(train, val)

				self.gp_split.shuffle()
				gallery = self.gp_split['gallery']
				probe = self.gp_split['probe']
				evaluation = self.model.evaluate(gallery, probe, evaluation, self._draw)

				if self.train_dir:
					self.model.reset()

				if run_once:
					break

		print("Final evaluation:")
		print(evaluation)
		if self.plot:
			plt.ioff()
			plt.show()

	def _read_samples(self, dir):
		return [
			Sample(os.path.join(dir, cls_dir, sample), self)
			for cls_dir in os.listdir(dir)
			if os.path.isdir(os.path.join(dir, cls_dir))
			for sample in os.listdir(os.path.join(dir, cls_dir))
			if os.path.isfile(os.path.join(dir, cls_dir, sample))
			and self.naming.valid(os.path.splitext(sample)[0])
		]

	def _group_samples(self, samples):
		f = (lambda sample: getattr(sample, self.grp_by)) if isinstance(self.grp_by, str) else self.grp_by
		bins = sorted(self.grp_bins or set(f(s) for s in samples))

		try:
			# Check if all bins are numbers
			all(bin < float('inf') for bin in bins)
			self.bin_labels = [
				((str(bins[i-1]) + " > ") if i > 0 else "") + '{}' + ((" <= " + str(bins[i])) if i < len(bins) else "")
				for i in range(len(bins) + 1)
			]
			return [[
				s for s in samples
				if (i == 0 or f(s) > bins[i-1])
				and (i == len(bins) or f(s) <= bins[i])
			] for i in range(len(bins) + 1)]
		except TypeError:
			self.bin_labels = bins
			return [[s for s in samples if f(s) == bin] for bin in bins]
