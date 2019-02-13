#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import os
import pickle
from tqdm import tqdm


def format_bin_label(x, _):
	return np.format_float_positional(x, precision=3, trim='-')


def f1_score(precision, recall):
	return 2 * precision * recall / (precision + recall)


if __name__ == '__main__':
	pred_dir = os.path.join('/hdd', 'EyeZ', 'Rot', 'Segmentation', 'Results')
	fig_file = os.path.join('/hdd', 'EyeZ', 'Rot', 'Segmentation', 'Results', 'Seg_ROC.eps')
	zoom_file = os.path.join('/hdd', 'EyeZ', 'Rot', 'Segmentation', 'Results', 'Seg_ROC_zoomed.eps')
	f1_file = os.path.join('/hdd', 'EyeZ', 'Rot', 'Segmentation', 'Results', 'Scores.txt')
	pr_file = os.path.join('/hdd', 'EyeZ', 'Rot', 'Segmentation', 'Results', '{}_precision_recall')
	load_pr = True

	plt.rcParams['font.family'] = 'Times New Roman'
	plt.rcParams['font.size'] = 30
	legend_size = 20
	color = iter(['orange', 'red', 'blue', 'green'])
	plt.figure('ROC')
	plt.figure('Zoomed ROC')

	with open(f1_file, 'w') as score:
		for model in ('RefineNet50', 'RefineNet101', 'SegNet', 'UNet'):
			print(model)
		
			if load_pr and os.path.isfile(pr_file.format(model)):
				with open(pr_file.format(model), 'rb') as f:
					precision, recall = pickle.load(f), pickle.load(f)
		
			else:
				dir = os.path.join(pred_dir, model.lower())
				gt = []
				pred = []
				for f in tqdm(os.listdir(dir)):
					if '_mask' in f:
						continue
					f = os.path.join(dir, f)
			
					p = plt.imread(f)
					m = 255 * plt.imread(os.path.splitext(f)[0] + '_mask.png')
					try:
						pred.extend(p.reshape(360*480))
					except ValueError:
						pred.extend(p[...,0].reshape(360*480))
					try:
						gt.extend(m.reshape(360*480))
					except ValueError:
						gt.extend(m[...,0].reshape(360*480))
		
				print("Generating curve")
				precision, recall, _ = precision_recall_curve(gt, pred)
		
				# Hack for outliers
				p_bad = np.where(np.sign(np.diff(precision)) == -1)[0]
				for bad in p_bad:
					precision[bad+1] = precision[bad]
				r_bad = np.where(np.sign(np.diff(recall)) == 1)[0]
				for bad in r_bad:
					recall[bad+1] = recall[bad]
				
				# Save precision and recall
				with open(pr_file.format(model), 'wb') as f:
					pickle.dump(precision, f)
					pickle.dump(recall, f)
			
			max_f1_score = (None, None, float('-inf'))
			for (p, r) in zip(precision, recall):
				f1 = f1_score(p, r)
				if f1 > max_f1_score[2]:
					max_f1_score = (p, r, f1)
			score.write(f"{model}: {max_f1_score}, AUC {auc(recall, precision)}\n")
		
			c = next(color)
			plt.figure('ROC')
			plt.plot(recall, precision, label=model, linewidth=2, color=c)
			plt.plot(max_f1_score[1], max_f1_score[0], 'o', color=c)
			plt.figure('Zoomed ROC')
			plt.plot(recall, precision, label=model, linewidth=2, color=c)
			plt.plot(max_f1_score[1], max_f1_score[0], 'o', markersize=15, color=c)
	
	plt.figure('ROC')
	plt.grid()
	plt.gca().xaxis.set_major_formatter(tick.FuncFormatter(format_bin_label))
	plt.gca().yaxis.set_major_formatter(tick.FuncFormatter(format_bin_label))
	plt.margins(0)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.xlim(0, 1.01)
	plt.ylim(0, 1.01)
	plt.xticks(np.linspace(0.2, 1, 5))
	plt.yticks(np.linspace(0, 1, 6))
	plt.legend(loc='lower left', fontsize=legend_size)
	plt.savefig(fig_file, bbox_inches='tight')
	plt.tight_layout(pad=0)
	
	plt.figure('Zoomed ROC')
	plt.grid()
	plt.gca().xaxis.set_major_formatter(tick.FuncFormatter(format_bin_label))
	plt.gca().yaxis.set_major_formatter(tick.FuncFormatter(format_bin_label))
	plt.margins(0)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.xlim(0.9, 1.01)
	plt.ylim(0.9, 1.01)
	plt.xticks(np.linspace(0.92, 1, 5))
	plt.yticks(np.linspace(0.9, 1, 6))
	plt.legend(loc='upper right', fontsize=legend_size, borderpad=0.2, borderaxespad=0.2, labelspacing=0.2)
	plt.savefig(zoom_file, bbox_inches='tight')
	plt.tight_layout(pad=0)
	
	plt.show()

