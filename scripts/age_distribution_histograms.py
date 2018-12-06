import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import operator
import os
import pickle
from scipy import stats
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import get_id_info


PLOT = True
SAVE = True
SAVE_PATH = os.path.join('/home', 'matej')
SAVE_NAME = 'Age_Hist_{}{}.eps'


def main():
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.rcParams['font.size'] = 30
	plt.rcParams['figure.figsize'] = (4, 8)

	ids = sorted(get_id_info(), key=operator.attrgetter('age'))
	ages = np.array([id.age for id in ids], dtype=int)
	m_ages = np.array([id.age for id in ids if id.gender == 'm'], dtype=int)
	f_ages = np.array([id.age for id in ids if id.gender == 'f'], dtype=int)

	ymax = plot(ages, 'Total', ages.min(), ages.max())
	plot(m_ages, 'Male', ages.min(), ages.max(), ymax)
	plot(f_ages, 'Female', ages.min(), ages.max(), ymax)
	if PLOT:
		plt.show()


def plot(ages, name, xmin, xmax, ymax=None):
	plt.figure(num=name)
	counts, bins, _ = plt.hist(ages, bins=10, color='blue', edgecolor='black', range=(xmin, xmax))
	kde = stats.gaussian_kde(ages)
	x = np.linspace(xmin, xmax, 1000)
	k = sum(count * bin_size for (count, bin_size) in zip(counts, np.diff(bins)))
	plt.plot(x, k * kde(x), color='red')
	
	if ymax is None:
		ymax = int(counts.max())

	plt.margins(0)
	plt.xticks(bins[::2])
	plt.yticks(np.linspace(0, ymax, 6, dtype=int))
	plt.gca().xaxis.set_major_formatter(tick.FuncFormatter(format_bin_label))
	plt.title(name)
	plt.xlabel("Age [Years]")
	plt.ylabel("Number of Subjects")
	plt.grid(axis='y')
	plt.tight_layout(pad=0)
	
	if SAVE:
		plt.savefig(os.path.join(SAVE_PATH, SAVE_NAME.format(name, '')))
		
		fig = plt.figure(num=name+'_No_Y')
		counts, bins, _ = plt.hist(ages, bins=10, color='blue', edgecolor='black', range=(xmin, xmax))
		kde = stats.gaussian_kde(ages)
		x = np.linspace(xmin, xmax, 1000)
		k = sum(count * bin_size for (count, bin_size) in zip(counts, np.diff(bins)))
		plt.plot(x, k * kde(x), color='red')
		
		plt.margins(0)
		plt.xticks(bins[::2])
		plt.yticks(np.linspace(0, ymax, 6, dtype=int), labels=[])
		plt.gca().xaxis.set_major_formatter(tick.FuncFormatter(format_bin_label))
		plt.title(name)
		plt.xlabel("Age [Years]")
		plt.grid(axis='y')
		plt.tight_layout(pad=0)
		plt.savefig(os.path.join(SAVE_PATH, SAVE_NAME.format(name, '_No_Y')))
		
	return ymax

	'''
	new_ax = plt.twinx()
	kde = stats.gaussian_kde(ages)
	x = np.linspace(ages.min(), ages.max(), 1000)
	new_ax.plot(x, kde(x), color='red')
	new_ax.yaxis.set_visible(False)

	plt.figure(num=name+'2')
	_, bins, _ = plt.hist(ages, bins=10, color='blue', edgecolor='black', density=True)
	plt.xticks(bins, fontsize=12)
	plt.gca().xaxis.set_major_formatter(tick.FuncFormatter(format_bin_label))
	plt.title(name, fontsize=20)
	plt.xlabel("Age [Years]", fontsize=16)
	plt.ylabel("Number of Subjects", fontsize=16)
	plt.margins(x=0)
	kde = stats.gaussian_kde(ages)
	x = np.linspace(ages.min(), ages.max(), 1000)
	plt.plot(x, kde(x), color='red')
	'''


def format_bin_label(x, pos):
	return np.format_float_positional(x, precision=3, trim='-')
	

if __name__ == '__main__':
	main()
