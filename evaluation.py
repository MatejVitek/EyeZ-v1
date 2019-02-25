import numpy as np
import sklearn.metrics as skmetrics


class Evaluation(object):
	def __init__(self):
		self._threshold = None
		self._far = None
		self._frr = None
		self._tar = None
		self.auc = Metric("AUC")
		self.eer = Metric("EER")
		self.ver1far = Metric("VER@1FAR")
		self.ver01far = Metric("VER@0.1FAR")

	def __str__(self):
		return "\n".join(str(metric) for metric in self.all_metrics())

	def all_metrics(self):
		return [metric for metric in self.__dict__.values() if isinstance(metric, Metric)]

	def compute_error_rates(self, dist_matrix, g_classes, p_classes, **kw):
		closest_only = kw.get('closest_only', False)
		imp_matrix = kw.get('impostor_matrix') or dist_matrix
		imp_classes = kw.get('impostor_classes') or p_classes
		n_points = kw.get('n_points', 5000)

		self._threshold = np.linspace(dist_matrix.min(), dist_matrix.max(), n_points)#np.unique(dist_matrix)

		same = self._same(dist_matrix, g_classes, p_classes, closest_only)
		diff = self._diff(imp_matrix, g_classes, imp_classes, closest_only)
		print(f"Client verification attempts: {len(same)}")
		print(f"Impostor verification attempts: {len(diff)}")

		self._far = np.array([np.count_nonzero(diff <= t) / len(diff) for t in self._threshold])
		self._frr = np.array([np.count_nonzero(same > t) / len(same) for t in self._threshold])
		self._tar = 1 - self._frr

		return self._far, self._frr, self._threshold

	def update_auc(self):
		auc = skmetrics.auc(self._far, self._tar)
		self.auc.update(auc)
		return auc

	def update_eer(self):
		eer_ = eer(self._far, self._frr, self._threshold)
		self.eer.update(eer_)
		return eer_

	def update_ver1far(self):
		ver = ver_at_far(self._far, self._tar, self._threshold, 0.01)
		self.ver1far.update(ver)
		return ver

	def update_ver01far(self):
		ver = ver_at_far(self._far, self._tar, self._threshold, 0.001)
		self.ver01far.update(ver)
		return ver

	@staticmethod
	def _same(dist_matrix, g_classes, p_classes, closest):
		if not closest:
			return np.array([d for ((g, p), d) in np.ndenumerate(dist_matrix) if g_classes[g] == p_classes[p]])
		return np.array([
			min(same)
			for same in (
				[d for g, d in enumerate(col) if g_classes[g] == p_classes[p]]
				for p, col in enumerate(dist_matrix.T)
			)
			if same
		])

	@staticmethod
	def _diff(dist_matrix, g_classes, p_classes, closest):
		if not closest:
			return np.array([d for ((g, p), d) in np.ndenumerate(dist_matrix) if g_classes[g] != p_classes[p]])
		return np.array([
			min(diff)
			for diff in (
				[d for g, d in enumerate(col) if g_classes[g] == diff_cls]
				for p, col in enumerate(dist_matrix.T)
				for diff_cls in set(g_classes) if diff_cls != p_classes[p]
			)
			if diff
		])


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
		if self.std:
			return f"{self.name} (\u03BC \u00B1 \u03C3): {self.mean} \u00B1 {self.std}"
		else:
			return f"{self.name}: {self.mean}"

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


def eer(far, frr, x):
	# See https://math.stackexchange.com/questions/2987246/finding-the-y-coordinate-of-the-intersection-of-two-functions-when-all-x-coordin
	# and https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line for explanation of below formulas
	try:
		i = np.argwhere(np.diff(np.sign(far - frr))).flatten()[0]
	except IndexError:
		# No intersection
		return 1.

	x = (x[i], x[i+1])
	y = (far[i], far[i+1], frr[i], frr[i+1])
	return (
		((x[0] * y[1] - x[1] * y[0]) * (y[2] - y[3]) - (x[0] * y[3] - x[1] * y[2]) * (y[0] - y[1])) /
		((x[0] - x[1]) * (-y[0] + y[1] + y[2] - y[3]))
	)


def ver_at_far(far, tar, x, far_point=0.01):
	try:
		i = np.argwhere(np.diff(np.sign(far - far_point))).flatten()[0]
	except IndexError:
		# No intersection
		return 0.

	alpha = (far_point - x[i]) / (x[i+1] - x[1])
	return alpha * tar[i] + (1 - alpha) * tar[i+1]
