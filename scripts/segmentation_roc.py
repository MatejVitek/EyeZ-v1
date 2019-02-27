#!/usr/bin/env python3
import itertools
from joblib import Parallel, delayed
import matplotlib.cm
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import pickle
from random import shuffle
from scipy.interpolate import interp1d
from sklearn.metrics import precision_recall_curve, auc
import sys
from tqdm import tqdm, trange

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import get_eyez_dir


# Bootstrapping parameters
K = 1
ratio = 1

# Which config to use
cfg = 'sclera'

# Should precision/recall be loaded from above file(s) (True) or should it be computed anew (False)
load_pr = True

seg_results = os.path.join(get_eyez_dir(), 'Segmentation', 'Results')
if cfg == 'vessels':
	models = ('Miura_MC', 'Miura_RLT', 'Miura_MC_norm', 'Miura_RLT_norm', 'agt', 'segnet')
	pred_dir = os.path.join(seg_results, 'Vessels')
	fig_file = os.path.join(pred_dir, 'Vessels_ROC.eps')
	zoom_file = os.path.join(pred_dir, 'Vessels_ROC_Zoomed.eps')
	f1_file = os.path.join(pred_dir, 'Vessels_Scores.txt')
	pr_file = os.path.join(pred_dir, '{}_precision_recall')
	#colors = iter([hsv_to_rgb((h, 1, 1)) for h in np.linspace(0, 1, len(models), endpoint=False)])
	#colors = iter(matplotlib.cm.get_cmap('plasma')(np.linspace(0, 1, len(models), endpoint=False)))
	colors = iter(['red', 'blue', 'green', 'black'])
	legend_loc = 'upper right'
elif cfg == 'sclera':
	models = ('RefineNet-50', 'RefineNet-101', 'UNet', 'SegNet')
	pred_dir = os.path.join(seg_results, 'Sclera')
	fig_file = os.path.join(pred_dir, 'Sclera_ROC.eps')
	zoom_file = os.path.join(pred_dir, 'Sclera_ROC_Zoomed.eps')
	f1_file = os.path.join(pred_dir, 'Sclera_Scores.txt')
	pr_file = os.path.join(pred_dir, '{}_precision_recall')
	#colors = iter([hsv_to_rgb((h, 1, 1)) for h in np.linspace(0, 1, len(models), endpoint=False)])
	#colors = iter(matplotlib.cm.get_cmap('plasma')(np.linspace(0, 1, len(models), endpoint=False)))
	colors = iter(['red', 'blue', 'green', 'purple'])
	legend_loc = 'lower left'

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['font.size'] = 22
legend_size = 20
plt.figure('ROC')
plt.figure('Zoomed ROC')


class FileInfo(object):
	def __init__(self, model, file, precision, recall, threshold):
		self.model = model
		self.file = file
		self.precision = precision
		self.recall = recall
		self.threshold = np.append(threshold, [1.])


def _process_file(dir, model, f):
	if '_gt' in f:
		return None
	f = os.path.join(dir, model, f)
	if not os.path.isfile(f):
		return None
	gt_f = os.path.splitext(f)[0] + '_gt.png'
	if not os.path.isfile(gt_f):
		return None

	p = plt.imread(f)
	if p.ndim > 2:
		p = rgb2gray(p)
	m = plt.imread(gt_f)
	if m.ndim > 2:
		m = rgb2gray(m)
	# Binarise mask
	m = m.round().astype(int)
	if p.shape != m.shape:
		raise ValueError(f"Different dimensions ({p.shape} and {m.shape}) for base and GT mask in image {f}.")

	return FileInfo(model, f, *precision_recall_curve(m.flatten(), p.flatten()))


def format_bin_label(x, _):
	return np.format_float_positional(x, precision=3, trim='-')


def f1_score(precision, recall):
	if precision == recall == 0:
		return 0
	return 2 * precision * recall / (precision + recall)

def find_max_f1(precision_v, recall_v):
	max_f1 = (None, None, float('-inf'))
	for (p, r) in zip(precision_v, recall_v):
		f1 = f1_score(p, r)
		if f1 > max_f1[2]:
			max_f1 = (p, r, f1)
	return max_f1


def rgb2gray(rgba):
	return np.dot(rgba[...,:3], [0.2989, 0.587, 0.114])


if __name__ == '__main__':
	info = {model: [] for model in models}

	if load_pr:
		print("Loading existing precision/recall info")
		existing = {model for model in models if os.path.isfile(pr_file.format(model))}
		for model in existing:
			with open(pr_file.format(model), 'rb') as f:
				info[model] = pickle.load(f)
		models = list(set(models) - existing)

	print("Computing precision/recall")
	output = [x for x in Parallel(n_jobs=-1, backend='multiprocessing')(
		delayed(_process_file)(pred_dir, model, file)
		for model, file in itertools.chain.from_iterable(itertools.product([model], os.listdir(os.path.join(pred_dir, model))) for model in models)
	) if x is not None]
	for file in output:
		info[file.model].append(file)

	print(info.keys())

	# Save precision and recall
	for model in models:
		with open(pr_file.format(model), 'wb') as f:
			pickle.dump(info[model], f)

	print("Computing means and stds")
	with open(f1_file, 'w') as score:
		for model in tqdm(info):
			pop_size = round(ratio * len(info[model]))
			interp_points = 1000
			max_f1 = np.empty((K, pop_size, 3))
			auc_ = np.empty((K, pop_size))
			precision = np.empty((K, pop_size, interp_points))
			recall = np.empty((K, pop_size, interp_points))

			for k in trange(K):
				# Get a random sample population of size determined by ratio
				shuffle(info[model])
				files = info[model][:round(ratio * len(info[model]))]

				for i, file in tqdm(enumerate(files)):
					# Max F1 score and its corresponding precision/recall values
					max_f1[k, i, :] = find_max_f1(file.precision, file.recall)

					# AUC
					auc_[k, i] = auc(file.recall, file.precision)

					# Precision/recall interpolation
					precision[k, i, :] = interp1d(file.threshold, file.precision, fill_value='extrapolate')(np.linspace(0, 1, interp_points))
					recall[k, i, :] = interp1d(file.threshold, file.recall, fill_value='extrapolate')(np.linspace(0, 1, interp_points))

			# Compute mean and std of scores
			score.write(f"{model}: {max_f1.mean((0, 1))} \u00B1 {max_f1.std((0, 1))}, AUC {auc_.mean()} \u00B1 {auc_.std()}\n")

			# Plot mean and upper/lower std ROCs
			pr = {}
			for name, y in (('precision', precision), ('recall', recall)):
				mean, std = y.mean((0, 1)), y.std((0, 1))
				pr[name, 'mean'] = mean
				pr[name, 'std1'] = mean + std
				pr[name, 'std2'] = mean - std

			c = next(colors)
			plt.figure('ROC')
			plt.plot(pr['recall', 'mean'], pr['precision', 'mean'], label=model, linewidth=2, color=c)
			for std in ('std1', 'std2'):
				plt.plot(pr['recall', std], pr['precision', std], ':', linewidth=1, color=c)
			plt.figure('Zoomed ROC')
			plt.plot(pr['recall', 'mean'], pr['precision', 'mean'], label=model, linewidth=2, color=c)
			for std in ('std1', 'std2'):
				plt.plot(pr['recall', std], pr['precision', std], ':', linewidth=1, color=c)
			#plt.plot(max_f1.mean((0, 1))[1], max_f1.mean((0, 1))[0], 'o', markersize=15, color=c)	

			# Max f1 on mean plot
			max_f1_mean = find_max_f1(pr['precision', 'mean'], pr['recall', 'mean'])
			plt.plot(max_f1_mean[1], max_f1_mean[0], 'o', markersize=15, color=c)


	plt.figure('ROC')
	plt.grid()
	plt.gca().xaxis.set_major_formatter(FuncFormatter(format_bin_label))
	plt.gca().yaxis.set_major_formatter(FuncFormatter(format_bin_label))
	plt.margins(0)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.xlim(0, 1.01)
	plt.ylim(0, 1.01)
	plt.xticks(np.linspace(0.2, 1, 5))
	plt.yticks(np.linspace(0, 1, 6))
	plt.legend(loc=legend_loc, fontsize=legend_size)
	plt.savefig(fig_file, bbox_inches='tight')
	plt.tight_layout(pad=0)

	plt.figure('Zoomed ROC')
	plt.grid()
	plt.gca().xaxis.set_major_formatter(FuncFormatter(format_bin_label))
	plt.gca().yaxis.set_major_formatter(FuncFormatter(format_bin_label))
	plt.margins(0)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.xlim(0.9, 1.001)
	plt.ylim(0.9, 1.001)
	plt.xticks(np.linspace(0.92, 1, 5))
	plt.yticks(np.linspace(0.9, 1, 6))
	plt.legend(loc='upper right', fontsize=legend_size, borderpad=0.2, borderaxespad=0.2, labelspacing=0.2)
	plt.savefig(zoom_file, bbox_inches='tight')
	plt.tight_layout(pad=0)

	plt.show()


'''
The above bootstrapping computes mean and std over all pictures and all folds
The below bootstrapping only computes mean and std over folds, first computing the mean precision/recall of all pictures in population for every fold

if __name__ == '__main__':
	info = {model: [] for model in models}
	
	if load_pr:
		print("Loading existing precision/recall info")
		existing = {model for model in models if os.path.isfile(pr_file.format(model))}
		for model in existing:
			with open(pr_file.format(model), 'rb') as f:
				info[model] = pickle.load(f)
		models = list(set(models) - existing)
	
	print("Computing precision/recall")
	output = [x for x in Parallel(n_jobs=-1, backend='multiprocessing')(
		delayed(_process_file)(pred_dir, model, file)
		for model, file in itertools.chain.from_iterable(itertools.product([model], os.listdir(os.path.join(pred_dir, model))) for model in models)
	) if x is not None]
	for file in output:
		info[file.model].append(file)
	
	print(info.keys())
	
	# Save precision and recall
	for model in models:
		with open(pr_file.format(model), 'wb') as f:
			pickle.dump(info[model], f)

	print("Computing means and stds")
	with open(f1_file, 'w') as score:
		for model in tqdm(info):
			pop_size = round(ratio * len(info[model]))
			interp_points = 1000
			max_f1 = np.empty((K, 3))
			auc_ = np.empty(K)
			precision = np.empty((K, interp_points + 1))
			recall = np.empty((K, interp_points + 1))
			
			for k in trange(K):
				# Get a random sample population of size determined by ratio
				shuffle(info[model])
				files = info[model][:round(ratio * len(info[model]))]
				
				# Get mean precision/recall of population
				precision[k, :-1] = np.array([
					interp1d(file.threshold, file.precision, fill_value='extrapolate')(np.linspace(0, 1, interp_points))
					for file in files
				]).mean(0).clip(0., 1.)
				recall[k, :-1] = np.array([
					interp1d(file.threshold, file.recall, fill_value='extrapolate')(np.linspace(0, 1, interp_points))
					for file in files
				]).mean(0).clip(0., 1.)
				
				# Fix for bad recall extrapolation
				precision[k, -1] = precision[k, -2]
				recall[k, -1] = 0.
				
				# Max F1 score and its corresponding precision/recall values
				max_f1[k, :] = find_max_f1(precision[k, :], recall[k, :])
				# AUC
				auc_[k] = auc(recall[k, :], precision[k, :])
			
			# Compute mean and std of scores
			score.write(f"{model}: {max_f1.mean(0)} \u00B1 {max_f1.std(0)}, AUC {auc_.mean()} \u00B1 {auc_.std()}\n")
			
			# Plot mean and upper/lower std ROCs
			pr = {}
			for name, y in (('precision', precision), ('recall', recall)):
				mean, std = y.mean(0), y.std(0)
				pr[name, 'mean'] = mean
				pr[name, 'std1'] = mean + std
				pr[name, 'std2'] = mean - std
			
			c = next(colors)
			plt.figure('ROC')
			plt.plot(pr['recall', 'mean'], pr['precision', 'mean'], label=model, linewidth=2, color=c)
			for std in ('std1', 'std2'):
				plt.plot(pr['recall', std], pr['precision', std], ':', linewidth=1, color=c)
			plt.figure('Zoomed ROC')
			plt.plot(pr['recall', 'mean'], pr['precision', 'mean'], label=model, linewidth=2, color=c)
			for std in ('std1', 'std2'):
				plt.plot(pr['recall', std], pr['precision', std], ':', linewidth=1, color=c)
			#plt.plot(max_f1.mean(0)[1], max_f1.mean(0)[0], 'o', markersize=15, color=c)
			
			# Max f1 on mean plot
			max_f1_mean = find_max_f1(pr['precision', 'mean'], pr['recall', 'mean'])
			plt.plot(max_f1_mean[1], max_f1_mean[0], 'o', markersize=15, color=c)
'''
