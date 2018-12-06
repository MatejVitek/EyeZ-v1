import os

from keras.applications import ResNet50
from keras.models import load_model, Model
from keras.optimizers import RMSprop, SGD

from cross_validate import CV
import utils


# Recognition
DATA_DIR = os.path.join(utils.get_rot_dir(), 'Recognition', 'all_directions_same_id')
DATA = {'train': 'stage1', 'test': 'stage2'}


def main():
	first_unfreeze = None
	feature_size = None

	# ResNet50
	#model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	#first_unfreeze = 143
	#feature_size = 1024

	# ScleraNET
	model = load_model(os.path.join(utils.get_rot_dir(), 'Recognition', 'all_directions_same_id', 'models', 'id_dir_prediction.75-0.667.hdf5'))
	model = Model(model.input, [model.get_layer('dense_1').output])#'final_features').output])

	CV(
		model,
		None,#os.path.join(DATA_DIR, DATA['train']),
		os.path.join(DATA_DIR, DATA['test']),
		first_unfreeze=first_unfreeze,
		primary_epochs=30,
		secondary_epochs=10,
		opt1=RMSprop(lr=1e-4),
		opt2=SGD(lr=1e-5, momentum=0.5, nesterov=True),
		both_eyes_same_class=False,
		mirrored_offset=0,
		feature_size=feature_size,
		plot=True,
		naming=r'ie_d_n',
		directions=r'lrsu',
		eyes=r'LR',
		naming_strict=False,
		group_by=None,#'age',
		bins=(25, 40),
		interbin_evaluation=False
	)(
		k=2,
		gp_split=0.3
	)


main()
