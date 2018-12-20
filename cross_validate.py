import os

from data_split import CVSplit, RatioSplit
from model_wrapper import CVModel, TrainableNNModel
from plot import Painter
import utils


EXTRA_INFO = utils.get_id_info()


class CV(object):
	def __init__(self, model, train_dir, test_dir, **kw):
		"""
		Initializes a new cross-validation object

		:param CVModel model: Predictor wrapped for evaluation
		:param train_dir: Directory with training data (must adhere to keras structure). If None, no training will be done.
		:type  train_dir: str or None
		:param str test_dir: Directory with testing data (must adhere to keras structure)
		:param bool crossbin_eval: Whether to use cross-bin evaluation in impostor testing
		"""

		# Model and directories
		self.model = model
		self.train_dir = train_dir
		self.test_dir = test_dir

		# If training directory was specified, model has to be trainable (uncomment after CVModel done)
		if self.train_dir and not isinstance(self.model, TrainableNNModel):
			raise ValueError("If training, model must be a subclass of TrainableNNModel")
		
		# Other settings
		self.grp_cross_eval = kw.get('crossbin_eval')

		# Uninitialized variables
		self.tv_split = self.gp_split = None

	def __call__(self, *args, **kw):
		return self.cross_validate(*args, **kw)

	def cross_validate(self, k=10, gp_split=0.3, plot=True):
		"""
		Cross validate model

		:param k: Number of folds
		:param gp_split: Gallery/probe split ratio
		:param plot: Painter object to use or boolean value
		:type  plot: Painter or bool or None

		:return: Final evaluation
		:rtype:  Evaluation
		"""

		# Use default painter if unspecified
		new_painter = plot is True
		if new_painter:
			plot = Painter(k=k)
			plot.add_figure('EER', xlabel="Threshold", ylabel="FAR/FRR")
			plot.add_figure('ROC Curve', xlabel="FAR", ylabel="TAR")
			plot.init()

		# Special case for k = 1
		run_once = False
		if k <= 1:
			k = 2
			run_once = True

		# Group data (only one group if no grouping done)
		self.data = {t: [data] for (t, data) in self.data.items()}

		if self.train_dir:
			if self.grp_by:
				self.data['train'] = self._group_samples(self.data['train'][0])
		else:
			self.data['train'] = [None]

		if self.grp_by:
			self.data['test'] = self._group_samples(self.data['test'][0])

		if len(self.data['test']) == 1 < len(self.data['train']):
			self.data['test'] *= len(self.data['train'])
		elif len(self.data['train']) == 1 < len(self.data['test']):
			self.data['train'] *= len(self.data['test'])
		if not self.bin_labels:
			self.bin_labels = [None] * len(self.data['train'])
		assert len(self.data['train']) == len(self.data['test']) == len(self.bin_labels)

		evaluation = None
		for train_group, test_group, bin in zip(self.data['train'], self.data['test'], self.bin_labels):
			if self.grp_by:
				print(bin.format(self.grp_by if isinstance(self.grp_by, str) else "x") + ":")

			if self.train_dir:
				self.tv_split = CVSplit(train_group, k)
			self.gp_split = RatioSplit(test_group, gp_split)

			for fold in range(k):
				print(f"Fold {fold+1}:")

				if self.train_dir:
					assert isinstance(self.model, TrainableNNModel)
					train = self.tv_split[(x for x in range(len(self.tv_split)) if x != fold)]
					val = self.tv_split[fold]
					self.model.train(train, val)

				self.gp_split.shuffle()
				gallery = self.gp_split['gallery']
				probe = self.gp_split['probe']
				evaluation = self.model.evaluate(gallery, probe, evaluation, plot)

				if self.train_dir:
					self.model.reset()

				if plot:
					plot.next_color()

				if run_once:
					break

		print("Final evaluation:")
		print(evaluation)

		if new_painter:
			plot.finalize()

		return evaluation
