import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn.metrics
import os
import random

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import load_img
from keras.utils import Sequence, to_categorical


class CV(object):
	def __init__(self, base_model, train_dir, test_dir, first_unfreeze=0, primary_epochs=50, secondary_epochs=20, both_eyes_same_class=True, feature_size=1024, distance='cosine', plot=True, metric='AUC'):
		self.base = base_model
		self.first_unfreeze = first_unfreeze
		self.base_weights = self.base.get_weights()
		
		self.input_size = base_model.input_shape[1:3]
		if any(size is None for size in self.input_size):
			self.input_size = (256, 256)
		self.feature_size = feature_size
		self.batch_size = 32
		
		self.train_dir = train_dir
		self.test_dir = test_dir
		self.epochs1 = primary_epochs
		self.epochs2 = secondary_epochs
		self.eyes_same_class = both_eyes_same_class
		self.dist = distance
		self.plot = plot
		if self.plot:
			plt.ion()
		self.metric = metric
		
		self.model = None
		self.tv_split = self.gp_split = None
		self.t_gen = self.v_gen = None
		self.k = 0
		self.n_classes = 0
		
	def __call__(self, *args, **kwargs):
		return self.cross_validate(*args, **kwargs)

	def cross_validate(self, k=10, gp_split=0.3):
		self.k = k

		# Count classes
		self.n_classes = max(
			self.get_class_label(cls)
			for d in (self.train_dir, self.test_dir)
			for cls in os.listdir(d)
			if os.path.isdir(os.path.join(d, cls))
		) + 1
		print(f"Found {self.n_classes} classes.")

		self.tv_split = CVSplit(self.train_dir, self.k)
		self.gp_split = RatioSplit(self.test_dir, gp_split)
		
		AUC = np.empty(self.k)
		for i in range(self.k):
			print(f"Fold {i+1}:")
			
			self._build_model()
			
			self.t_gen = LabeledImageGenerator(
				self.tv_split[(x for x in range(self.k) if x != i)],
				self.get_class_label,
				self.n_classes,
				target_size=self.input_size,
				batch_size=self.batch_size
			)
			self.v_gen = LabeledImageGenerator(
				self.tv_split[i],
				self.get_class_label,
				self.n_classes,
				target_size=self.input_size,
				batch_size=self.batch_size
			)
		
			# Freeze base layers
			for layer in self.base.layers:
				layer.trainable = False
			print("Training top layers:")
			self._fit_model(epochs=self.epochs1, opt='rmsprop', loss='categorical_crossentropy')

			# Unfreeze the last few base layers
			for layer in self.base.layers[self.first_unfreeze:]:
				layer.trainable = True
			print("Training unfrozen layers:")
			self._fit_model(epochs=self.epochs2, opt=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
		
			# Remove softmax layer and evaluate model
			print("Evaluating model:")
			self._remove_top_from_model()
			AUC[i] = self._evaluate_model()
			print(f"AUC: {AUC[i]}")
		
			# Clean up
			for layer in self.base.layers:
				layer.trainable = True
			self.base.set_weights(self.base_weights)
			
		print(f"AUC (\u03BC \u00B1 \u03C3): {np.mean(AUC)} \u00B1 {np.std(AUC)}")
		if self.plot:
			plt.ioff()
			plt.show()
			
		return np.mean(AUC), np.std(AUC)
		
	def get_class_label(self, cls_name):
		if os.path.isfile(cls_name):
			cls_name = os.path.dirname(cls_name)
		if os.path.isdir(cls_name):
			cls_name = os.path.basename(cls_name)
		cls_name = cls_name.split('_')
		n = int(cls_name[0]) - 1
		if self.eyes_same_class:
			return n
		return 2 * n if cls_name[1] in ('L', 'l') else 2 * n + 1
	
	def _build_model(self):
		# Add own top layer(s)
		self.model = Sequential()
		self.model.add(self.base)
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
		
	def _fit_model(self, epochs, opt='SGD', loss='categorical_crossentropy'):
		self.model.compile(optimizer=opt, loss=loss)
		self.model.fit_generator(
			self.t_gen,
			epochs=epochs,
			validation_data=self.v_gen
		)
	
	def _evaluate_model(self):
		self.gp_split.shuffle()
		g_gen = ImageGenerator(
			self.gp_split['gallery'],
			target_size=self.input_size,
			batch_size=self.batch_size
		)
		p_gen = ImageGenerator(
			self.gp_split['probe'],
			target_size=self.input_size,
			batch_size=self.batch_size
		)
		
		# Get class labels
		g_classes = [self.get_class_label(g) for g in self.gp_split[0]]
		p_classes = [self.get_class_label(p) for p in self.gp_split[1]]
	
		# rows = samples, columns = features
		g_features = self.model.predict_generator(g_gen)
		p_features = self.model.predict_generator(p_gen)
				
		# rows = gallery, columns = probe
		dist_matrix = sp.spatial.distance.cdist(g_features, p_features, metric=self.dist)
		
		'''Testing code:
		g_classes = [0, 1, 2, 2]
		p_classes = [0, 0, 1, 1, 2, 2, 2]
		dist_matrix = np.empty((len(g_classes), len(p_classes)))
		for (i, row) in enumerate(dist_matrix):
			for (j, col) in enumerate(dist_matrix[i]):
				x = random.uniform(0, 0.3) if g_classes[i] == p_classes[j] else random.uniform(0.7, 1)
				if random.random() < 0.2:
					x = 1 - x
				dist_matrix[i, j] = x
		'''
		
		n_points = 100
		frr = np.empty((len(dist_matrix), n_points))
		far = np.empty((len(dist_matrix), n_points))
		threshold = np.linspace(0, 1, n_points)
		for (i, row) in enumerate(dist_matrix):
			same = np.array([x for (j, x) in enumerate(row) if g_classes[i] == p_classes[j]])
			diff = np.array([x for (j, x) in enumerate(row) if g_classes[i] != p_classes[j]])
			
			frr[i] = np.array([np.count_nonzero(same > t) / len(same) for t in threshold])
			far[i] = np.array([np.count_nonzero(diff <= t) / len(diff) for t in threshold])
		
		x = far.mean(axis=0)
		y = 1 - frr.mean(axis=0)
		if self.plot:
			self._draw(x, y, "FAR", "1 - FRR")
		if self.metric.lower() == 'auc':
			return sklearn.metrics.auc(x, y)
		
	def _draw(self, x, y, xlabel=None, ylabel=None):
		plt.plot(x, y)
		if xlabel:
			plt.xlabel(xlabel)
		if ylabel:
			plt.ylabel(ylabel)
		plt.draw()
		plt.pause(1)
		
		
class CVSplit(object):
	def __init__(self, dir, n_folds, all_dirs_same_fold=True):
		# Read samples
		samples = list({
			os.path.join(dir, cls_dir, self._get_image_name(sample, all_dirs_same_fold))
			for cls_dir in os.listdir(dir)
			if os.path.isdir(os.path.join(dir, cls_dir))
			for sample in os.listdir(os.path.join(dir, cls_dir))
			if os.path.isfile(os.path.join(dir, cls_dir, sample))
		})
			
		# Shuffle and split sample list
		random.shuffle(samples)
		self.folds = np.array_split(samples, n_folds)
		
		# Expand list of samples with different directions
		if all_dirs_same_fold:
			for (i, fold) in enumerate(self.folds):
				self.folds[i] = [
					sample.format(direction)
					for sample in fold
					for direction in 'lrsu'
					if os.path.isfile(sample.format(direction))
				]
		
		print(f"Found {sum(len(fold) for fold in self.folds)} images.")
		
	def _get_image_name(self, name, all_dirs_same_fold):
		if not all_dirs_same_fold:
			return name
		name = name.split('_')
		name[1] = '{}'
		return '_'.join(name)
		
	def __getitem__(self, index):
		try:
			# Return a single fold
			return self.folds[index]
		except TypeError:
			# Return multiple folds as a single flattened list
			return list(itertools.chain.from_iterable(self.folds[i] for i in index))


class RatioSplit(object):
	def __init__(self, dir, ratio):
		self.samples = [
			os.path.join(dir, cls_dir, s)
			for cls_dir in os.listdir(dir) if os.path.isdir(os.path.join(dir, cls_dir))
			for s in os.listdir(os.path.join(dir, cls_dir))
		]
		self.split = round(ratio * len(self.samples))
		
		print(f"Found {len(self.samples)} images. Placed {self.split} in gallery and {len(self.samples) - self.split} in probe.")
	
	def __getitem__(self, index):
		if index == 0 or isinstance(index, str) and index.lower() in ('g', 'gallery'):
			return self.samples[:self.split]
		elif index == 1 or isinstance(index, str) and index.lower() in ('p', 'probe'):
			return self.samples[self.split:]
		else:
			raise IndexError("The only allowed indices are 0/'g'/'gallery' and 1/'p'/'probe'.")
			
	def shuffle(self):
		return random.shuffle(self.samples)


class ImageGenerator(Sequence):
	def __init__(self, data, target_size=(256, 256), batch_size=32):
		self.data = data
		self.target_size = target_size
		self.batch_size = batch_size
		self.batch = None
		self.on_epoch_end()
		
	def __len__(self):
		return int(np.ceil(len(self.data) / self.batch_size))
		
	def __getitem__(self, index):
		self._init_batch(min(self.batch_size, len(self.data) - index * self.batch_size))
		for (i, sample) in enumerate(self.data[index * self.batch_size:(index + 1) * self.batch_size]):
			self._read_sample(i, sample)
		return self.batch
			
	def _init_batch(self, size):
		self.batch = np.empty((size, *self.target_size, 3), dtype=float)
	
	def _read_sample(self, i, sample):
		self.batch[i] = load_img(sample, target_size=self.target_size)
		
	def on_epoch_end(self):
		random.shuffle(self.data)
		

class LabeledImageGenerator(ImageGenerator):
	def __init__(self, data, class_f, n_classes, *args, **kwargs):
		super().__init__(data, *args, **kwargs)
		self.class_f = class_f
		self.n_classes = n_classes

	def _init_batch(self, size):
		self.batch = (
			np.empty((size, *self.target_size, 3), dtype=float),
			np.empty((size, self.n_classes), dtype=int)
		)

	def _read_sample(self, i, sample):
		self.batch[0][i] = load_img(sample, target_size=self.target_size)
		self.batch[1][i] = to_categorical(self.class_f(sample), num_classes=self.n_classes, dtype=int)

