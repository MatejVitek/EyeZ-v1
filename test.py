import matplotlib.pyplot as plt
import numpy as np
import random
import sklearn.metrics

from cross_validate import *


def main():
	#test_roc()
	test_eer()
	#test_eval()
	#test_metric()


def test_roc():
	g_classes = [random.randint(0, 10) for _ in range(100)]
	p_classes = [random.randint(0, 10) for _ in range(300)]
	dist_matrix = np.empty((len(g_classes), len(p_classes)))
	for (i, row) in enumerate(dist_matrix):
		for (j, col) in enumerate(dist_matrix[i]):
			x = random.uniform(0, 0.5) if g_classes[i] == p_classes[j] else random.uniform(0.5, 1)
			if random.random() < 0.1:
				x = 1 - x
			dist_matrix[i, j] = x
	x, y = CV._error_rates(dist_matrix, g_classes, p_classes, sample_thresholding=True)
	y = 1 - y
	print(sklearn.metrics.auc(x, y))
	plt.plot(x, y)
	plt.show()
	
	
def test_eer(n_points=1000):
	threshold = np.linspace(0, 1, n_points)
	far = np.sort(np.random.rand(n_points))
	frr = np.sort(np.random.rand(n_points))[::-1]
	print(CV._compute_eer(far, frr))
	plt.plot(threshold, far)
	plt.plot(threshold, frr)
	plt.show()


def test_eval(n_data=1000):
	data = np.random.rand(n_data, 2)
	print(f"{data[:,0].mean()} \u00B1 {data[:,0].std()}")
	print(f"{data[:,1].mean()} \u00B1 {data[:,1].std()}")
	
	e = Evaluation()
	e.auc.update(data[:,0])
	e.eer.update(data[:,1])
	print(e)
	
	
def test_metric(n_data=1000):
	data = np.random.rand(n_data)
	print(f"{data.mean()} \u00B1 {data.std()}")
	
	e = Metric(data)
	print(e)


if __name__ == '__main__':
	main()
