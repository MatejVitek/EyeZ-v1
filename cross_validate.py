from collections import defaultdict

from dataset import Dataset, CVSplit, RatioSplit
from model_wrapper import CVModel, TrainableNNModel
from plot import Painter
import utils


EXTRA_INFO = utils.get_id_info()


class CV(object):
	def __init__(self, model):
		"""
		Initializes a new cross-validation object

		:param CVModel model: Predictor wrapped for evaluation
		"""

		self.model = model

	def __call__(self, train, test, *args, **kw):
		if isinstance(test, dict):
			return self.cross_validate_grouped(train, test, *args, **kw)
		else:
			return self.cross_validate(train, test, *args, **kw)

	def cross_validate(self, train, test, k=10, plot=True, evaluation=None, **kw):
		"""
		Cross validate model

		:param train: Dataset to train on. If None, only evaluation will be done.
		:type  train: Dataset or CVSplit or None
		:param test: Dataset to test on
		:type  test: Dataset or GPSplit
		:param int k: Number of folds
		:param plot: Painter object to use or boolean value
		:type  plot: Painter or bool or None
		:param evaluation: If specified, will use this as the pre-existing evaluation
		:type  evaluation: Evaluation or None
		:param kw: Additional arguments to pass to :py:CVModel.evaluate

		:return: Final evaluation
		:rtype:  Evaluation
		"""

		# If training dataset was specified, model has to be trainable
		if train and not isinstance(self.model, TrainableNNModel):
			raise TypeError("If training, model must be a subclass of TrainableNNModel")

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

		# If train is passed as a Dataset, split it into k folds
		if isinstance(train, Dataset):
			train = CVSplit(train, k)

		# If test is passed as a Dataset, split it into gallery and probe as 30:70
		if isinstance(test, Dataset):
			test = RatioSplit(test, 0.3)

		for fold in range(k):
			print(f"Fold {fold+1}:")

			if train:
				train_data = train[(x for x in range(len(train)) if x != fold)]
				val_data = train[fold]
				self.model.train(train_data, val_data)

			test.new_split()
			evaluation = self.model.evaluate(test.gallery, test.probe, evaluation=evaluation, plot=plot, **kw)

			if train:
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

	def cross_validate_grouped(self, train, test, *args, **kw):
		"""
		Cross validate a grouped dataset

		:param train: Dictionary of training groups. If None, no training will be done.
		:param test: Dictionary of testing groups. If train was specified, both should be of the same length.
		:param args: Additional args to pass to :py:cross_validate
		:param evaluation: If specified, will use this as the pre-existing evaluation. Cannot be used with return_separate.
		:type  evaluation: Evaluation or None
		:param bool return_separate: Whether to combine the result into one evaluation or return a dictionary
		:param bool intergroup_evaluation: Whether to use samples from different groups for impostor testing
		:param kw: Additional keyword args to pass to :py:cross_validate

		:return: Final evaluation(s)
		:rtype:  Evaluation
		"""

		return_separate = kw.pop('return_separate', False)
		evaluation = kw.pop('evaluation', {} if return_separate else None)
		inter_eval = kw.pop('intergroup_evaluation', False)

		if return_separate and evaluation:
			raise ValueError("return_separate and evaluation are mutually exclusive")

		if not train:
			train = defaultdict()

		for label in test:
			print(label)
			if return_separate:
				evaluation[label] = self.cross_validate(train[label], test[label], *args, evaluation=None, **kw)
			else:
				evaluation = self.cross_validate(train[label], test[label], *args, evaluation=evaluation, **kw)

		return evaluation
