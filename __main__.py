import os

from keras.applications import ResNet50
from keras.models import load_model, Model
from keras.optimizers import RMSprop

from cross_validate import CV
from dataset import Dataset
from dist_models import *
from model_wrapper import *
from naming import NamingParser
from plot import Painter
import utils


K = 10


# Should dataset be grouped by an attribute (such as age)? If not, set GROUP_BY to None.
GROUP_BY = None
#GROUP_BY = 'age'
# Bins to group into. For more info, see dataset.Dataset.group_by.
BINS = (25, 40)
# Are we using intergroup evaluation? See cross_validate.CV.cross_validate_grouped.
INTERGROUP = True


# Do we want a plot of our models' performances?
PLOT = False


# Training and testing datasets. If no training is to be done, set train to None.
DATA_DIR = os.path.join(utils.get_rot_dir(), 'Recognition', 'all_directions_same_id')
DATA = {'train': None, 'test': 'stage2'}


def main():
	# Define file naming rules
	naming = NamingParser(
		r'ie_d_n',
		eyes=r'LR',
		directions=r'lrsu',
		strict=True
	)
	both_eyes_same_class = False
	mirrored_offset = 0

	train = None
	if DATA['train']:
		train = Dataset(
			os.path.join(DATA_DIR, DATA['train']),
			naming=naming,
			both_eyes_same_class=both_eyes_same_class,
			mirrored_offset=mirrored_offset
		)
		if GROUP_BY:
			train = train.group_by(GROUP_BY, BINS)
	test = Dataset(
		os.path.join(DATA_DIR, DATA['test']),
		naming=naming,
		both_eyes_same_class=both_eyes_same_class,
		mirrored_offset=mirrored_offset
	)
	if GROUP_BY:
		test = test.group_by(GROUP_BY, BINS)

	models = (correlation(),)

	painter = None
	if PLOT:
		painter = Painter(
			lim=(0, 1.01),
			xticks=np.linspace(0.2, 1, 5),
			yticks=np.linspace(0, 1, 6),
			#colors=['r', 'b', 'g'],
			k=len(test) * K * len(models) if GROUP_BY else K * len(models),
			labels=(
				[f"{key} (k = {k})" for key in test.keys() for k in range(K)] if GROUP_BY and K > 1
				else [f"{key}" for key in test.keys()] if GROUP_BY
				else [f"k = {k}" for k in range(K)]
			)
			#labels=["CNN", "SIFT"]
		)
		painter.add_figure('EER', xlabel='Threshold', ylabel='FAR/FRR')
		painter.add_figure('ROC Curve', xlabel='FAR', ylabel='TAR', save='Sclera-ROC.eps')
		painter.init()

	for model in models:
		evaluation = CV(model)(
			train,
			test,
			K,
			plot=painter,
			closest_only=True,
			intergroup_evaluation=INTERGROUP
		)
		if GROUP_BY:
			for k, v in evaluation.items():
				print(f'{k}:\n{str(v)}\n')
		else:
			print(evaluation)


# Configs
BATCH_SIZE = 32
DIST = 'cosine'


def base_nn_config(model, train=False, feature_size=None, first_unfreeze=None):
	if train:
		return TrainablePredictorModel(
			model,
			primary_epochs=30,
			secondary_epochs=10,
			feature_size=feature_size,
			first_unfreeze=first_unfreeze,
			primary_opt=RMSprop(lr=1e-4),
			secondary_opt=SGD(lr=1e-5, momentum=0.5, nesterov=True),
			batch_size=BATCH_SIZE,
			distance=DIST
		)
	else:
		return PredictorModel(model, batch_size=BATCH_SIZE, distance=DIST)


def resnet50(train=True):
	return base_nn_config(
		ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg'),
		train=train,
		feature_size=1024,
		first_unfreeze=143
	)


def scleranet(layer='final_features'):
	model = load_model(os.path.join(utils.get_rot_dir(), 'Recognition', 'all_directions_same_id', 'models', 'id_dir_prediction.75-0.667.hdf5'))
	return base_nn_config(Model(model.input, [model.get_layer(layer).output]))


def descriptor(*args, **kw):
	return DirectDistanceModel(DescriptorModel(*args, **kw))


def correlation():
	return DirectDistanceModel(CorrelationModel())


main()
