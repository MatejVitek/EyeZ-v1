import os

from keras.applications import ResNet50
from keras.models import load_model, Model
from keras.optimizers import RMSprop

from cross_validate import CV
from dataset import Dataset
from dist_models import *
from hog import HOGModel
from model_wrapper import *
from naming import NamingParser
from plot import Painter, exp_format
import utils


K = 1


# Should dataset be grouped by an attribute (such as age)? If not, set GROUP_BY to None.
GROUP_BY = None
#GROUP_BY = 'age'
# Bins to group into. Ignored if GROUP_BY is None. For more info, see dataset.Dataset.group_by.
BINS = (25, 40)
# Are we using intergroup evaluation? Ignored if GROUP_BY is None. See cross_validate.CV.cross_validate_grouped.
INTERGROUP = True


# Do we want a plot of our models' performances?
PLOT = True
# Do we want to save the evaluation results?
SAVE = False


# Training and testing datasets. If no training is to be done, set train to None.
DATA_DIR = os.path.join(utils.get_eyez_dir(), 'Recognition', 'Databases', 'Rot ScleraNet')
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

	models = (scleranet(), descriptor('sift'), descriptor('surf'), descriptor('orb'), descriptor('sift', True))
	labels = ("CNN", "SIFT", "SURF", "ORB", "dSIFT")
	results_dir = os.path.join(utils.get_eyez_dir(), 'Recognition', 'Results')
	results = os.path.join(results_dir, f'{DATA["test"]}.txt')

	painter = None
	if PLOT:
		painter = Painter(
			lim=(0, 1.01),
			xticks=np.linspace(0.2, 1, 5),
			yticks=np.linspace(0, 1, 6),
			colors=['r', 'b', 'g', 'purple', 'black'],
			k=len(test) * K * len(models) if GROUP_BY else K * len(models),
			#labels=(
			#	[f"{key} (k = {k})" for key in test.keys() for k in range(K)] if GROUP_BY and K > 1
			#	else [f"{key}" for key in test.keys()] if GROUP_BY
			#	else [f"k = {k}" for k in range(K)]
			#)
			labels=labels,
			font='Times New Roman',
			font_size=22,
			legend_size=20
		)
		painter.init()
		painter.add_figure('EER', xlabel='Threshold', ylabel='FAR/FRR')
		painter.add_figure(
			'ROC Curve', save=os.path.join(results_dir, f'Sclera-ROC-{DATA["test"]}.eps'),
			xlabel='FAR', ylabel='TAR', legend_loc='lower right'
		)
		painter.add_figure(
			'Semilog ROC Curve', save=os.path.join(results_dir, f'Sclera-ROC-log-{DATA["test"]}.eps'),
			xlabel='FAR', ylabel='TAR', legend_loc='best',
			xscale='log', xlim=(1e-3, 1.01), xticks=(1e-3, 1e-2, 1e-1, 1), x_tick_formatter=exp_format
		)

	results_file = None
	if SAVE:
		results_file = open(results, 'w')

	try:
		for model, label in zip(models, labels):
			if SAVE:
				results_file.write(f"{label}:\n\n")
			evaluation = CV(model)(
				train,
				test,
				K,
				plot=painter,
				closest_only=True,
				intergroup_evaluation=INTERGROUP,
				save=os.path.join(utils.get_eyez_dir(), 'Recognition', 'Results', f'{label}_{DATA["test"]}.pkl'),
				use_precomputed=True
			)
			if GROUP_BY:
				for k, v in evaluation.items():
					print(f"{k}:\n{str(v)}\n")
					if SAVE:
						results_file.write(f"{k}:\n{str(v)}\n")
			else:
				print(evaluation)
				if SAVE:
					results_file.write(str(evaluation))
			if SAVE:
				results_file.write("\n\n\n\n")

	finally:
		if PLOT:
			painter.finalize()
		if SAVE:
			results_file.close()


# Configs
BATCH_SIZE = 32
DIST = 'cosine'


def base_nn_config(model, train=False, feature_size=None, first_unfreeze=None):
	if train:
		return TrainablePredictorModel(
			model,
			primary_epochs=50,
			secondary_epochs=30,
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
	model = load_model(os.path.join(utils.get_eyez_dir(), 'Recognition', 'Models', 'Rot ScleraNet', 'id_dir_prediction.75-0.667.hdf5'))
	return base_nn_config(Model(model.input, [model.get_layer(layer).output]))


def descriptor(*args, **kw):
	return DirectDistanceModel(DescriptorModel(*args, **kw))


def hog(*args, **kw):
	return PredictorModel(HOGModel(*args, **kw), batch_size=BATCH_SIZE, distance=DIST)


# Too slow and doesn't work
def correlation():
	return DirectDistanceModel(CorrelationModel())


#DATA_DIR = os.path.join(utils.get_eyez_dir(), 'Segmentation', 'Results', 'Vessels')
#for data_dir in ('Miura_MC_norm',):
#	DATA = {'train': None, 'test': data_dir}
#	main()


main()
