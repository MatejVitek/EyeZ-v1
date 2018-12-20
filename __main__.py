import os

from keras.applications import ResNet50
from keras.models import load_model, Model
from keras.optimizers import RMSprop, SGD

from cross_validate import CV
from model_wrapper import *
from plot import Painter
import utils


K = 1


# Recognition
DATA_DIR = os.path.join(utils.get_rot_dir(), 'Recognition', 'all_directions_same_id')
DATA = {'train': 'stage1', 'test': 'stage2'}
MODEL = 'ResNet'


def main():
	with Painter(k=2*K, lim=(0, 1.01), xticks=np.linspace(0.2, 1, 5), yticks=np.linspace(0, 1, 6)) as painter:
		painter.add_figure('EER', xlabel='Threshold', ylabel='FAR/FRR')
		painter.add_figure('ROC Curve', xlabel='FAR', ylabel='TAR')
		for layer in 'dense_1', 'final_features':
			model = scleranet(layer)

			CV(
				model,
				os.path.join(DATA_DIR, DATA['train']) if isinstance(model, TrainableNNModel) else None,
				os.path.join(DATA_DIR, DATA['test']),
				both_eyes_same_class=False,
				mirrored_offset=0,
				naming=r'ie_d_n',
				directions=r'lrsu',
				eyes=r'LR',
				naming_strict=False,
				group_by=None,#'age',
				bins=(25, 40),
				interbin_evaluation=False
			)(
				k=K,
				gp_split=0.3,
				plot=painter
			)


# Configs
BATCH_SIZE = 32
DIST = 'cosine'


def base_config(model, train=False, feature_size=None, first_unfreeze=None):
	if train:
		return TrainableNNModel(
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
		return NNModel(model, batch_size=BATCH_SIZE, distance=DIST)


def resnet50(train=True):
	return base_config(
		ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg'),
		train=train,
		feature_size=1024,
		first_unfreeze=143
	)


def scleranet(layer='dense_1'):
	model = load_model(os.path.join(utils.get_rot_dir(), 'Recognition', 'all_directions_same_id', 'models', 'id_dir_prediction.75-0.667.hdf5'))
	return base_config(Model(model.input, [model.get_layer(layer).output]))


main()
