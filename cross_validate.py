import itertools
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn.metrics
import os
import random

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image
from keras.utils import Sequence, to_categorical


class CV(object):
	def __init__(self, base_model, train_dir, test_dir, **kwargs):
		self.base = base_model
		self.base_weights = self.base.get_weights()
		self.input_size = base_model.input_shape[1:3]
		if any(size is None for size in self.input_size):
			self.input_size = (256, 256)
			
		self.train_dir = train_dir
		self.test_dir = test_dir
		
		self.batch_size = kwargs.get('batch_size', 32)
		self.dist = kwargs.get('distance', 'cosine')
		self.epochs1 = kwargs.get('primary_epochs') or kwargs.get('top_epochs') or kwargs.get('epochs1') or kwargs.get('epochs', 50)
		self.epochs2 = kwargs.get('secondary_epochs') or kwargs.get('unfrozen_epochs') or kwargs.get('epochs2', 20)
		self.eyes_same_class = kwargs.get('both_eyes_same_class', True)
		self.feature_size = kwargs.get('feature_size', 1024)
		self.first_unfreeze = kwargs.get('first_unfreeze')
		self.opt1 = kwargs.get('opt1') or kwargs.get('opt', 'rmsprop')
		self.opt2 = kwargs.get('opt2', SGD(lr=0.0001, momentum=0.9))
		self.plot = kwargs.get('plot', True)
		
		self.model = None
		self.tv_split = self.gp_split = None
		self.n_classes = 0
		self.color = None

	def __call__(self, *args, **kwargs):
		return self.cross_validate(*args, **kwargs)

	def cross_validate(self, k=10, gp_split=0.3):
		if self.plot:
			plt.ion()
			colors = np.vstack((np.linspace(0, 1, k), [1] * k, [1] * k)).T

		run_once = False
		if k <= 1:
			k = 2
			run_once = True

		# Count classes
		self.n_classes = max(
			self.get_class_label(cls)
			for d in (self.train_dir, self.test_dir)
			for cls in os.listdir(d)
			if os.path.isdir(os.path.join(d, cls))
		) + 1
		print(f"Found {self.n_classes} classes.")

		self.tv_split = CVSplit(self.train_dir, k)
		self.gp_split = RatioSplit(self.test_dir, gp_split)

		evaluation = Evaluation()
		for i in range(k):
			print(f"Fold {i+1}:")

			if self.plot:
				self.color = matplotlib.colors.hsv_to_rgb(colors[i])

			# Build and train the model
			self._build_model()
			self._train_model(i)
		
			# Remove softmax layer and evaluate model
			self._remove_top_from_model()
			self._evaluate_model(evaluation)
		
			# Clean up
			for layer in self.base.layers:
				layer.trainable = True
			self.base.set_weights(self.base_weights)
			
			if run_once:
				break
		
		print("Final evaluation:")
		print(evaluation)
		if self.plot:
			plt.ioff()
			plt.show()
		
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
		
	def _train_model(self, step):
		t_gen = LabeledImageGenerator(
			self.tv_split[(x for x in range(len(self.tv_split)) if x != step)],
			self.get_class_label,
			self.n_classes,
			target_size=self.input_size,
			batch_size=self.batch_size
		)
		v_gen = LabeledImageGenerator(
			self.tv_split[step],
			self.get_class_label,
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
	
	def _evaluate_model(self, evaluation):
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
		
		# Get class labels
		g_classes = [self.get_class_label(g) for g in self.gp_split['gallery']]
		p_classes = [self.get_class_label(p) for p in self.gp_split['probe']]
	
		# rows = samples, columns = features
		print("Predicting gallery features:")
		g_features = self.model.predict_generator(g_gen, verbose=1)
		print("Predicting probe features:")
		p_features = self.model.predict_generator(p_gen, verbose=1)
		
		same = np.array([
			(g, p)
			for (i, g) in enumerate(g_features)
			for (j, p) in enumerate(p_features)
			if g_classes[i] == p_classes[j]
		])
		diff = np.array([
			(g, p)
			for (i, g) in enumerate(g_features)
			for (j, p) in enumerate(p_features)
			if g_classes[i] != p_classes[j]
		])
						
		# rows = gallery, columns = probe
		dist_matrix = np.absolute(sp.spatial.distance.cdist(g_features, p_features, metric=self.dist))

		# Get FAR and FRR
		far, frr, threshold = self._error_rates(dist_matrix, g_classes, p_classes, return_threshold=True, n_points=1000)
		
		# EER
		eer = self._compute_eer(far, frr)
		print(f"EER: {eer}")
		evaluation.eer.update(eer)
		if self.plot:
			self._draw(threshold, far, "Threshold", "FAR", figure="EER")
			self._draw(threshold, frr, "Threshold", "FRR", figure="EER")
		
		# AUC
		trr = 1 - frr
		auc = sklearn.metrics.auc(far, trr)
		print(f"AUC: {auc}")
		evaluation.auc.update(auc)
		if self.plot:
			self._draw(far, trr, "FAR", "1 - FRR", figure="ROC Curve")
	
	@staticmethod
	def _error_rates(dist_matrix, g_classes, p_classes, return_threshold=False, n_points=0):
		if n_points:
			threshold = np.linspace(dist_matrix.min(), dist_matrix.max(), n_points)
		else:
			threshold = np.unique(dist_matrix)
		
		same = np.array([d for ((g, p), d) in np.ndenumerate(dist_matrix) if g_classes[g] == p_classes[p]])
		diff = np.array([d for ((g, p), d) in np.ndenumerate(dist_matrix) if g_classes[g] != p_classes[p]])
			
		far = np.array([np.count_nonzero(diff <= t) / len(diff) for t in threshold])
		frr = np.array([np.count_nonzero(same > t) / len(same) for t in threshold])
		
		if return_threshold:
			return far, frr, threshold
		else:
			return far, frr
		
	@staticmethod
	def _compute_eer(far, frr):
		# See https://math.stackexchange.com/questions/2987246/finding-the-y-coordinate-of-the-intersection-of-two-functions-when-all-x-coordin for explanation of below formulas
		i = np.argwhere(np.diff(np.sign(far - frr))).flatten()[0]
		return (far[i] * frr[i+1] - far[i+1] * frr[i]) / (far[i] - far[i+1] - frr[i] + frr[i+1])
		
	def _draw(self, x, y, xlabel=None, ylabel=None, figure=None, clear=False):
		plt.figure(num=figure, clear=clear)
		plt.plot(x, y, color=self.color)
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
			
	def __len__(self):
		return len(self.folds)


class RatioSplit(object):
	def __init__(self, dir, ratio):
		self.samples = [
			os.path.join(dir, cls_dir, s)
			for cls_dir in os.listdir(dir) if os.path.isdir(os.path.join(dir, cls_dir))
			for s in os.listdir(os.path.join(dir, cls_dir))
		]
		self.split = round(ratio * len(self.samples))
		
		print(f"Found {len(self.samples)} images. Placing {self.split} in gallery and {len(self.samples) - self.split} in probe.")
	
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
	def __init__(self, data, target_size=(256, 256), batch_size=32, shuffle=True):
		self.data = data
		self.target_size = target_size
		self.batch_size = batch_size
		self.batch = None
		self.shuffle = shuffle
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
		self.batch[i] = self._load_img(sample)
		
	def _load_img(self, sample):
		x = image.load_img(sample, target_size=self.target_size)
		return image.img_to_array(x) / 255
		
	def on_epoch_end(self):
		if self.shuffle:
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
		self.batch[0][i] = self._load_img(sample)
		self.batch[1][i] = to_categorical(self.class_f(sample), num_classes=self.n_classes, dtype=int)
		
		
class Evaluation(object):
	def __init__(self):
		self.auc = Metric("AUC")
		self.eer = Metric("EER")
		
	def __str__(self):
		return "\n".join(str(metric) for metric in self.__dict__.values())
		

class Metric(object):
	def __init__(self, name, values=None, ddof=0):
		self.name = name
		
		self.mean = 0
		self.var = 0
		self.std = 0
		
		self._n = 0
		self._s = 0
		self._ddof = ddof
		
		if values is not None:
			self.update(values)
		
	def __str__(self):
		return f"{self.name} (\u03BC \u00B1 \u03C3): {self.mean} \u00B1 {self.std}"
		
	def __len__(self):
		return self._n
		
	def update(self, values):
		try:
			for v in values:
				self._update(v)
		except TypeError:
			self._update(values)
		
	def _update(self, value):
		self._n += 1
		
		old_mean = self.mean
		self.mean += (value - old_mean) / self._n
		
		self._s += (value - old_mean) * (value - self.mean)
		self.var = self._s / (self._n - self._ddof) if self._n > self._ddof else 0
		self.std = np.sqrt(self.var)

