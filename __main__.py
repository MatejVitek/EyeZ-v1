import numpy as np
import os

from keras.applications import ResNet50
from keras.models import load_model, Model
from keras.optimizers import RMSprop, SGD

from cross_validate import CV
from dataset import Dataset
from model_wrapper import *
from naming import NamingParser
from plot import Painter
import utils


K = 1
GROUP_BY = 'age'
BINS = (25, 40)
INTERGROUP = True


# Recognition
DATA_DIR = os.path.join(utils.get_rot_dir(), 'Recognition', 'all_directions_same_id')
DATA = {'train': None, 'test': 'stage2'}


def main():
	naming = NamingParser(
		r'ie_d_n',
		eyes=r'LR',
		directions=r'lrsu',
		strict=False
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

	painter = Painter(
		lim=(0, 1.01),
		xticks=np.linspace(0.2, 1, 5),
		yticks=np.linspace(0, 1, 6),
		colors=['r', 'g', 'b'],
		#k=(len(test) * K if GROUP_BY else K),
		labels=([f'{key}' for key in test.keys()] if GROUP_BY else [k for k in range(K)])
	)
	painter.add_figure('EER', xlabel='Threshold', ylabel='FAR/FRR')
	painter.add_figure('ROC Curve', xlabel='FAR', ylabel='TAR', save='Sclera-ROC2.eps')

	with painter:
		model = scleranet()
		evaluation = CV(model)(
			train,
			test,
			K,
			plot=painter,
			closest_only=True,
			intergroup_evaluation=True
		)
		if GROUP_BY:
			for k, v in evaluation.items():
				print(f'{k}:\n{str(v)}\n')
		else:
			print(evaluation)


# Configs
BATCH_SIZE = 32
DIST = 'cosine'


def base_config(model, train=False, feature_size=None, first_unfreeze=None):
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
	return base_config(
		ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg'),
		train=train,
		feature_size=1024,
		first_unfreeze=143
	)


def scleranet(layer='final_features'):
	model = load_model(os.path.join(utils.get_rot_dir(), 'Recognition', 'all_directions_same_id', 'models', 'id_dir_prediction.75-0.667.hdf5'))
	return base_config(Model(model.input, [model.get_layer(layer).output]))


main()
