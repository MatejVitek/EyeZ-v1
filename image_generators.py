import numpy as np
import random

from keras.preprocessing import image
from keras.utils import Sequence, to_categorical


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
		self.batch[i] = self._load_img(sample.file)

	def _load_img(self, f):
		x = image.load_img(f, target_size=self.target_size)
		return image.img_to_array(x) / 255

	def on_epoch_end(self):
		if self.shuffle:
			random.shuffle(self.data)


class LabeledImageGenerator(ImageGenerator):
	def __init__(self, data, n_classes, *args, **kw):
		super().__init__(data, *args, **kw)
		self.n_classes = n_classes

	def _init_batch(self, size):
		self.batch = (
			np.empty((size, *self.target_size, 3), dtype=float),
			np.empty((size, self.n_classes), dtype=int)
		)

	def _read_sample(self, i, sample):
		self.batch[0][i] = self._load_img(sample.file)
		self.batch[1][i] = to_categorical(sample.label, num_classes=self.n_classes, dtype=int)
