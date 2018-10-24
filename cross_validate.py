import numpy as np
import scipy as sp
import os
import random

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.preprocessing.image import load_img


class CV(object):
	def __init__(self, base_model, train_dir, test_dir, first_unfreeze=0, feature_size=1024, distance='cosine'):
		self.base = base_model
		self.first_unfreeze = first_unfreeze
		self.base_weights = self.base.get_weights()
		
		self.input_size = base_model.input_shape[1:3]
		if any(size is None for size in self.input_size):
			self.input_size = (256, 256)
		self.feature_size=feature_size
		self.batch_size = 32
		
		self.model = None
		self.dist = distance
		
		#self.tv_gen = TVGenerator(train_dir, k)
		self.gp_gen = GPGenerator(test_dir)
		self.k = 10
		self.gp_split = .3
		
	def __call__(self, *args, **kwargs):
		return self.cross_validate(*args, **kwargs)

	def cross_validate(self, k=None, gp_split=None):
		if k is not None:
			self.k = k
		if gp_split is not None:
			self.gp_split = gp_split
			
		for i in range(self.k):
			print(f"Fold number {i+1}:")
			
			self._build_model()

			# Freeze base layers
			for layer in self.base.layers:
				layer.trainable = False
				
			self._fit_model(epochs=30, opt='rmsprop', loss='categorical_crossentropy')

			# Unfreeze the last few base layers
			for layer in self.base.layers[self.first_unfreeze:]:
				layer.trainable = True
				
			self._fit_model(epochs=10, opt=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
		
			# Clean up
			for layer in self.base.layers:
				layer.trainable = True
			self.base.set_weights(self.base_weights)
		
			self._remove_top_from_model()
		self._evaluate_model()
		
	
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
			self.tv_gen.n_classes,
			name='top_softmax',
			activation='softmax'
		))
		
	def _remove_top_from_model(self):
		self.model.pop()
		
	def _fit_model(self, epochs, opt='SGD', loss='categorical_crossentropy'):
		t_gen = self.tv_gen.flow_from_directory(
			[x for x in range(k) if x != i],
			target_size=self.input_size,
			batch_size=self.batch_size
		)
		v_gen = self.tv_gen.flow_from_directory(
			[i],
			target_size=self.input_size,
			batch_size=self.batch_size
		)

		self.model.compile(optimizer=opt, loss=loss)
		self.model.fit_generator(
			t_gen,
			steps_per_epoch=int((self.k - 1) / self.k * self.tv_gen.n_samples / self.batch_size),
			epochs=epochs,
			validation_data=v_gen,
			validation_steps=int(1 / self.k * self.tv_gen.n_samples / self.batch_size)
		)
	
	def _evaluate_model(self):
		gallery, probe = self.gp_gen.new_split(self.gp_split)
		
		g_gen = self.gp_gen.flow_from_directory(
			gallery=True,
			target_size=self.input_size,
			batch_size=self.batch_size
		)
		p_gen = self.gp_gen.flow_from_directory(
			gallery=False,
			target_size=self.input_size,
			batch_size=self.batch_size
		)
	
		# TODO: remove below self.model lines
		# rows = samples, columns = features
		self.model = Sequential()
		self.model.predict_generator = lambda *_, **__: np.random.rand(len(gallery), self.feature_size)
		g_features = self.model.predict_generator(g_gen, steps=len(gallery))
		self.model.predict_generator = lambda *_, **__: np.random.rand(len(probe), self.feature_size)
		p_features = self.model.predict_generator(p_gen, steps=len(probe))
				
		# rows = gallery, columns = probe
		dist_matrix = sp.spatial.distance.cdist(g_features, p_features, metric=self.dist)
		
		n_points = 100
		frr = np.empty((len(dist_matrix), n_points))
		far = np.empty((len(dist_matrix), n_points))
		for (i, row) in enumerate(dist_matrix):
			same = np.array([x for (j, x) in enumerate(row) if gallery[i][0] == probe[j][0]])
			diff = np.array([x for (j, x) in enumerate(row) if gallery[i][0] != probe[j][0]])
			
			frr[i] = np.array([np.count_nonzero(same > threshold) / len(same) for threshold in np.linspace(0, 1, n_points)])
			far[i] = np.array([np.count_nonzero(diff <= threshold) / len(diff) for threshold in np.linspace(0, 1, n_points)])
		
		for row in frr:
			for e in row:
				if 0 < e < 1:
					print(e, end=' ')
			print()
		for row in far:
			for e in row:
				if 0 < e < 1:
					print(e, end=' ')
			print()
			

class TVGenerator(object):
	def __init__(self, dir, n_folds):
		indices = set()
		self.n_classes = 0
		self.n_samples = 0
		
		for cls in os.listdir(dir):
			cls_dir = os.path.join(dir, cls)
			if not os.path.isdir(cls_dir):
				continue
				
			samples = os.listdir(cls_dir)
			self.n_classes += 1
			self.n_samples += len(samples)
			
			indices.update(self._image_name(cls_dir, s) for s in samples)
			
		print(f"Found {self.n_samples} images belonging to {self.n_classes} classes.")
		
		indices = list(indices)
		random.shuffle(indices)

		self.folds = np.array_split(indices, n_folds)
		map(np.random.shuffle, self.folds)
			
	@staticmethod
	def _image_name(cls, name):
		name = name.split('_')
		name[1] = '{}'
		name = '_'.join(name)
		return os.path.join(cls, name)

	def flow_from_directory(self, fold_idx, target_size=(256, 256), batch_size=32):
		batch_x = np.empty((batch_size,) + target_size + (3,))
		batch_y = np.empty(batch_size, dtype=str)
		i = 0
		
		for f in fold_idx:
			for sample in self.folds[f]:
				for direction in 'lrsu':
					try:
						batch_x[i] = load_img(sample.format(direction), target_size=target_size)
						batch_y[i] = os.path.basename(os.path.dirname(sample))
					except FileNotFoundError:
						continue
					i += 1
					
					if i >= batch_size:
						yield batch_x, batch_y
						i = 0
						batch_x = np.empty((batch_size,) + target_size + (3,))
						batch_y = np.empty(batch_size, dtype=str)
						
		if i > 0:
			yield batch_x[:i], batch_y[:i]
		return
		
class GPGenerator(object):
	def __init__(self, dir):
		self.dir = dir
		self.files = [
			(os.path.basename(root), f)
			for (root, _, filenames) in os.walk(self.dir)
			for f in filenames
		]
		self.gallery = None
		self.probe = None
		
	def new_split(self, split=0.3):
		random.shuffle(self.files)
		split = int(split * len(self.files))
		self.gallery = sorted(self.files[:split])
		self.probe = sorted(self.files[split:])
		return self.gallery, self.probe
		
	def flow_from_directory(self, gallery=True, target_size=(256, 256), batch_size=32):
		source = self.gallery if gallery else self.probe
		batch = np.empty((batch_size, *target_size, 3))
		i = 0
		
		for sample in source:
			batch[i] = load_img(os.path.join(self.dir, *sample), target_size=target_size)
			i += 1
			if i >= batch_size:
				yield batch
				i = 0
				batch = np.empty((batch_size, *target_size, 3))
		
		if i > 0:
			yield batch[:i]
		return

