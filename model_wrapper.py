from abc import ABC, abstractmethod

from keras.optimizers import SGD, RMSprop


class CVModel(ABC):
	@abstractmethod
	def evaluate(self, data, evaluation=None):
		pass


class NNModel(CVModel):
	def __init__(self, model, **kw):
		"""
		NN model wrapper for cross-validation

		:param model: Model to evaluate (should not include the softmax layer)
		:param int batch_size: Batch size
		:param distance: Distance metric for evaluation (passed to scipy.spacial.distance.cdist)

		"""

		# Base model settings
		self.model = model
		self.input_size = self.model.input_shape[1:3]
		if any(size is None for size in self.input_size):
			self.input_size = (256, 256)

		# Other settings
		self.batch_size = kw.get('batch_size', 32)
		self.dist = kw.get('distance', 'cosine')

	def evaluate(self, data, evaluation=None):
		pass


class TrainableNNModel(NNModel):
	def __init__(self, model, **kw):
		"""
		:param primary_epochs: Epochs to train custom top layers only
		:param secondary_epochs: Epochs to train unfrozen layers and top layer
		:param int feature_size: Feature size of custom feature layer. If 0, no custom feature layer will be added.
		:param first_unfreeze: Index of the first layer to unfreeze in secondary training. If None, skip primary training and train the whole model.
		:type first_unfreeze: int or None
		:param opt1: Optimizer to use in primary training
		:param opt2: Optimizer to use in secondary training
		"""

		super().__init__(model, **kw)
		self.epochs1 = kw.get('primary_epochs') or kw.get('top_epochs') or kw.get('epochs1') or kw.get('epochs', 50)
		self.epochs2 = kw.get('secondary_epochs') or kw.get('unfrozen_epochs') or kw.get('epochs2', 20)
		self.feature_size = kw.get('feature_size', 1024)
		self.first_unfreeze = kw.get('first_unfreeze')
		self.opt1 = kw.get('opt1') or kw.get('opt', 'rmsprop')
		self.opt2 = kw.get('opt2', SGD(lr=0.0001, momentum=0.9))

	def train(self, data):
		pass
