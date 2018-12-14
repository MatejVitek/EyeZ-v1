L = 0
R = 1
C = 2
U = 3


import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import sklearn.metrics

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop

from data_split import CVSplit, RatioSplit
import evaluation
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

		self.label = None


class CV(object):
	def __init__(self, model, train_dir, test_dir, **kw):
		"""
		Initializes a new cross-validation object

		:param CVModel model: Predictor wrapped for evaluation
		:param train_dir: Directory with training data (must adhere to keras structure). If None, no training will be done.
		:type train_dir: str or None
		:param str test_dir: Directory with testing data (must adhere to keras structure)
		:param bool eyes_same_class: Whether both eyes should be counted as the same class.
		:param int mirrored_offset: Offset for mirrored identities. If 0, mirrored eyes will be counted as distinct classes.
		:param bool plot: Whether to draw plots
		:param group_by: Characteristic to group identities by. If None, no grouping will be done.
		:param bins: If sequence of n numbers [x1, ..., xn], will split data into n+1 bins (x<=x1, x1<x<=x2, ...).
		             Otherwise will split the data into n bins (x=x1, x=x2, ...).
		:param bool crossbin_eval: Whether to use cross-bin evaluation in impostor testing
		"""

		# Model and directories
		self.model = model
		self.train_dir = train_dir
		self.test_dir = test_dir

		# If training directory was specified, model has to be trainable (uncomment after CVModel done)
		#if self.train_dir and not isinstance(self.model, TrainableNNModel):
			#raise ValueError("If training, model must be a subclass of TrainableNNModel")

		# Delete after CVModel done
		self.base = model
		self.base_weights = self.base.get_weights()
		self.input_size = self.base.input_shape[1:3]
		if any(size is None for size in self.input_size):
			self.input_size = (256, 256)
		self.dist = kw.get('distance', 'cosine')
		self.batch_size = kw.get('batch_size', 32)
		self.epochs1 = kw.get('primary_epochs') or kw.get('top_epochs') or kw.get('epochs1') or kw.get('epochs', 50)
		self.epochs2 = kw.get('secondary_epochs') or kw.get('unfrozen_epochs') or kw.get('epochs2', 20)
		self.feature_size = kw.get('feature_size', 1024)
		self.first_unfreeze = kw.get('first_unfreeze')
		self.opt1 = kw.get('opt1') or kw.get('opt', 'rmsprop')
		self.opt2 = kw.get('opt2', SGD(lr=0.0001, momentum=0.9))
		
		# Other settings
		eyes_same_class = kw.get('both_eyes_same_class', True)
		mirrored_offset = kw.get('mirrored_offset', True)
		self.plot = kw.get('plot', True)
		self.grp_by = kw.get('group_by')
		self.grp_bins = kw.get('bins')
		self.grp_cross_eval = kw.get('crossbin_eval')

		# Uninitialized variables
		self.model = None
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

		eval_ = evaluation.Evaluation()
		for train_group, test_group, bin in zip(self.data['train'], self.data['test'], self.bin_labels):
			if self.grp_by:
				print(bin.format(self.grp_by if isinstance(self.grp_by, str) else "x") + ":")

			if self.train_dir:
				self.tv_split = CVSplit(train_group, k)
			self.gp_split = RatioSplit(test_group, gp_split)

			for i in range(k):
				print(f"Fold {i+1}:")

				if self.plot:
					self.color = colors[i]

				if self.train_dir:
					# Build and train the model
					self._build_model()
					self._train_model(i)

					# Remove softmax layer and evaluate the model
					self._remove_top_from_model()
					self._evaluate_model(eval_)

					# Clean up
					for layer in self.base.layers:
						layer.trainable = True
					self.base.set_weights(self.base_weights)

				else:
					self.model = self.base
					self._evaluate_model(eval_)

				if run_once:
					break

		print("Final evaluation:")
		print(eval_)
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
		
	def _build_model(self):
		# Add own top layer(s)
		self.model = Sequential()
		self.model.add(self.base)
		if self.feature_size:
			self.model.add(Dense(
				self.feature_size,
				name='top_fc',
				activation='relu'
			))
		self.model.add(Dense(
			self.n_classes,
			name='top_softmax',
			activation='softmax'
		))
		
	def _remove_top_from_model(self):
		self.model.pop()
		
	def _train_model(self, fold):
		t_gen = LabeledImageGenerator(
			self.tv_split[(x for x in range(len(self.tv_split)) if x != fold)],
			self.n_classes,
			target_size=self.input_size,
			batch_size=self.batch_size
		)
		v_gen = LabeledImageGenerator(
			self.tv_split[fold],
			self.n_classes,
			target_size=self.input_size,
			batch_size=self.batch_size
		)
	
		if self.first_unfreeze is not None:
			# Freeze base layers
			for layer in self.base.layers:
				layer.trainable = False
			print("Training top layers:")
			self._fit_model(t_gen, v_gen, epochs=self.epochs1, opt=self.opt1, loss='categorical_crossentropy')

			# Unfreeze the last few base layers
			for layer in self.base.layers[self.first_unfreeze:]:
				layer.trainable = True
			print("Training unfrozen layers:")
			self._fit_model(t_gen, v_gen, epochs=self.epochs2, opt=self.opt2, loss='categorical_crossentropy')

		else:
			print("Training model:")
			self._fit_model(t_gen, v_gen, epochs=self.epochs1, opt=self.opt1, loss='categorical_crossentropy')
		
	def _fit_model(self, t_gen, v_gen, epochs, opt='SGD', loss='categorical_crossentropy'):
		self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
		self.model.fit_generator(
			t_gen,
			epochs=epochs,
			validation_data=v_gen
		)
	
	def _evaluate_model(self, eval_):
		self.gp_split.shuffle()
		g_gen = ImageGenerator(
			self.gp_split['gallery'],
			target_size=self.input_size,
			batch_size=self.batch_size,
			shuffle=False
		)
		p_gen = ImageGenerator(
			self.gp_split['probe'],
			target_size=self.input_size,
			batch_size=self.batch_size,
			shuffle=False
		)
	
		# rows = samples, columns = features
		print("Predicting gallery features:")
		g_features = self.model.predict_generator(g_gen, verbose=1)
		print("Predicting probe features:")
		p_features = self.model.predict_generator(p_gen, verbose=1)
		
		g_classes = [s.label for s in self.gp_split['gallery']]
		p_classes = [s.label for s in self.gp_split['probe']]

		# rows = gallery, columns = probe
		dist_matrix = np.absolute(sp.spatial.distance.cdist(g_features, p_features, metric=self.dist))

		# Get FAR and FRR
		threshold = np.linspace(dist_matrix.min(), dist_matrix.max(), 1000)#np.unique(dist_matrix)
		far, frr = evaluation.error_rates(dist_matrix, g_classes, p_classes, threshold)
		
		# EER
		eer = eval_.eer(far, frr, threshold)
		print(f"EER: {eer}")
		if self.plot:
			self._draw(threshold, far, "Threshold", "FAR", figure="EER")
			self._draw(threshold, frr, "Threshold", "FRR", figure="EER")
		
		# AUC
		tar = 1 - frr
		auc = sklearn.metrics.auc(far, tar)
		print(f"AUC: {auc}")
		evaluation.auc.update(auc)
		if self.plot:
			self._draw(far, tar, "FAR", "TAR", figure="ROC Curve")

		# VER@1FAR
		ver1far = evaluation.ver_at_far(far, tar, threshold, 0.01)
		print(f"VER@1FAR: {ver1far}")
		evaluation.ver1far.update(ver1far)

	@staticmethod
	def _draw(self, x, y, xlabel=None, ylabel=None, figure=None, clear=False):
		plt.figure(num=figure, clear=clear)
		plt.plot(x, y, color=self.color)
		if xlabel:
			plt.xlabel(xlabel)
		if ylabel:
			plt.ylabel(ylabel)
		plt.draw()
		plt.pause(1)
