import cv2
import numpy as np
from tqdm import tqdm


class HOGModel:
	def __init__(self, input_shape=(256, 256), *args, **kw):
		self.input_shape = input_shape
		self.hog = cv2.HOGDescriptor(*args, **kw)

	def predict_generator(self, gen, verbose):
		if verbose:
			gen = tqdm(gen)
		return np.concatenate(np.array([self.predict_on_batch(batch) for batch in gen]), 1)

	def predict_on_batch(self, batch):
		return np.concatenate(np.array([self.predict(img) for img in batch]), 1)

	def predict(self, img):
		img = (img * 255).astype(int)
		return self.hog.compute(img)
