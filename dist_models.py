from abc import ABC, abstractmethod
from collections import OrderedDict
import cv2


class DistModel(ABC):
	def __init__(self, cache_size=100):
		self.size = cache_size
		if self.size < 2:
			raise ValueError("cache_size needs to be at least 2")
		self.cache = OrderedDict()

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
		print(f)
		img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
		return cv2.resize(img, (256, 256))

	@abstractmethod
	def _dist(self, cached1, cached2):
		pass


class SIFT(DistModel):
	def __init__(self, alg='sift', *args, cache_size=1000, **kw):
		super().__init__(cache_size)

		if alg.lower() == 'sift':
			self.alg = cv2.xfeatures2d.SIFT_create(*args, **kw)
		elif alg.lower() == 'surf':
			self.alg = cv2.xfeatures2d.SURF_create(*args, **kw)
		elif alg.lower() == 'orb':
			self.alg = cv2.ORB_create(*args, **kw)
		else:
			raise ValueError(f"Unsupported descriptor algorithm {sift}.")

		self.matcher = cv2.BFMatcher()

	def _cache_value(self, f):
		img = super()._cache_value(f)
		return self.alg.detectAndCompute(img, None)[1]

	def _dist(self, des1, des2):
		matches = self.matcher.knnMatch(des1, des2, k=2)
		good = [m1 for m1, m2 in matches if m1.distance < 0.75 * m2.distance]
		#return sum(m.distance for m in good) / len(good) if good else 1.
		return 1 - (len(good) / len(matches))
