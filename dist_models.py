from abc import ABC, abstractmethod
from collections import OrderedDict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


class DistModel(ABC):
	def __init__(self, cache_size=100, image_size=(256, 256)):
		self.size = cache_size
		if self.size < 2:
			raise ValueError("cache_size needs to be at least 2")
		self.cache = OrderedDict()
		self.image_size = image_size

	def distance(self, sample1, sample2):
		img = [None, None]
		for i, s in enumerate((sample1, sample2)):
			if s.basename not in self.cache:
				self.cache[s.basename] = self._cache_value(s.file)
			img[i] = self.cache[s.basename]
			if len(self.cache) > self.size:
				self.cache.popitem(False)

		return self._dist(*img)

	def _cache_value(self, f):
		img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
		return cv2.resize(img, self.image_size)

	@abstractmethod
	def _dist(self, cached1, cached2):
		pass


class DescriptorModel(DistModel):
	def __init__(self, descriptor='sift', dense=False, *args, cache_size=1000, **kw):
		super().__init__(cache_size=cache_size)

		#pylint: disable=no-member
		if descriptor.lower() == 'sift':
			self.alg = cv2.xfeatures2d.SIFT_create(*args, **kw)
		elif descriptor.lower() == 'surf':
			self.alg = cv2.xfeatures2d.SURF_create(*args, **kw)
		elif descriptor.lower() == 'orb':
			self.alg = cv2.ORB_create(*args, **kw)
		else:
			raise ValueError(f"Unsupported descriptor algorithm {descriptor}.")

		self.dense = dense
		self.matcher = cv2.BFMatcher()

	def _cache_value(self, f):
		img = super()._cache_value(f)
		if not self.dense:
			return self.alg.detectAndCompute(img, None)[1]
		step = 10 if self.dense is True else self.dense
		kp = [cv2.KeyPoint(x, y, step) for y in range(0, img.shape[0], step) for x in range(0, img.shape[1], step)]
		return self.alg.compute(img, kp)[1]

	def _dist(self, des1, des2):
		if des1 is None or len(des1) == 0 or des2 is None or len(des2) == 0:
			return 1.
		matches = self.matcher.knnMatch(des1, des2, k=2)
		good = [m[0] for m in matches if len(m) > 1 and m[0].distance < 0.75 * m[1].distance]
		#return sum(m.distance for m in good) / len(good) if good else 1.
		return 1 - (len(good) / len(matches))


# Too slow and doesn't work properly
class CorrelationModel(DistModel):
	def __init__(self, *args, cache_size=500, image_size=(128, 128), **kw):
		super().__init__(*args, cache_size=cache_size, image_size=image_size, **kw)

	def _dist(self, cached1, cached2):
		correlation = scipy.signal.correlate2d(cached1, cached2)
		f, axes = plt.subplots(2, 2)
		f.set_size_inches(15, 15)
		axes[0,0].imshow(cached1, cmap='gray')
		axes[0,1].imshow(cached2, cmap='gray')
		axes[1,0].imshow(correlation, cmap='gray')
		axes[1,1].axis('off')
		plt.show()
		print(correlation.max())
		return correlation.max() / max(np.linalg.norm(cached1), np.linalg.norm(cached2))
