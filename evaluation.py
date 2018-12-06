import numpy as np


class Evaluation(object):
	def __init__(self):
		self.auc = Metric("AUC")
		self.eer = Metric("EER")
		self.ver1far = Metric("VER@1FAR")

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
