import os

from keras.applications import ResNet50
from keras.layers import Input
from keras.models import load_model, Model
from keras.optimizers import RMSprop, SGD

from cross_validate import CV
from dataset import Dataset, L, R, C, U
from dist_models import *
from hog import HOGModel
from model_wrapper import *
from naming import NamingParser
from plot import Painter, exp_format
from utils import get_eyez_dir


# Configuration

config = 'compute'
config = 'plot'


# Configuration settings

# How many folds?
K = 10
# Do we want a plot of our models' performances?
PLOT = False
# Do we want to save the evaluation results to a text file?
SAVE = True
# Do we want to load precomputed distance matrices?
LOAD = False
# Plot size - whole column-width ('large') or half-column ('small')
SIZE = 'small'
#SIZE = 'large'

if config == 'plot':
	K = 1
	PLOT = True
	SAVE = False
	LOAD = True


# Special protocol settings

# Number of view directions in base template for each identity (or list of specific directions)
BASE_DIRS = 4
#BASE_DIRS = (C,)

# Image size to use in model evaluation (None to use default settings)
IMG_SIZE = None
#IMG_SIZE = (400, 400)

# Should dataset be grouped by an attribute (such as age)? If not, set GROUP_BY to None.
GROUP_BY = None
#GROUP_BY = 'age'
#GROUP_BY = 'gender'
# Bins to group into. Ignored if GROUP_BY is None. For more info, see dataset.Dataset.group_by.
BINS = (25, 40)
#BINS = None
# Are we using intergroup evaluation? Ignored if GROUP_BY is None. See cross_validate.CV.cross_validate_grouped.
INTERGROUP = False

# Ignore this
I = 0


# Training and testing datasets. If no training is to be done, set train to None.
DATA_DIR = os.path.join(get_eyez_dir(), 'Recognition', 'Databases', 'Rot ScleraNet')
DATA = {'train': None, 'test': 'stage2'}
#DATA_DIR = os.path.join(get_eyez_dir(), 'Recognition', 'Databases')
#DATA = {'train': None, 'test': 'SBVPI Scleras'}


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

	if IMG_SIZE:
		models = (
			scleranet(image_size=IMG_SIZE),
			*(descriptor(name, image_size=IMG_SIZE) for name in ('sift', 'surf', 'orb')),
			descriptor('sift', True, image_size=IMG_SIZE)
		)
	else:
		models = (scleranet(), *(descriptor(name) for name in ('sift', 'surf', 'orb')), descriptor('sift', True))
	labels = ("CNN", "SIFT", "SURF", "ORB", "dSIFT")

	# This is for plotting grouped, comment out otherwise
	#models = (models[I],)
	#labels = (labels[I],)

	res_path = [DATA['test']]
	if GROUP_BY:
		res_path.append(f'{GROUP_BY}{"_intergroup" if INTERGROUP else ""}')
	try:
		if BASE_DIRS < 4:
			res_path.append(f'{BASE_DIRS} direction{"s" if BASE_DIRS > 1 else ""} in base')
	except TypeError:
		dirs = {L: 'left', R: 'right', C: 'center', U: 'up'}
		res_path.append(f'{", ".join(dirs[d] for d in sorted(BASE_DIRS))} in base')
	res_dir = os.path.join(get_eyez_dir(), 'Recognition', 'Results', *res_path)
	os.makedirs(res_dir, exist_ok=True)
	
	group_suffix = '_{group}' if GROUP_BY else ''
	fold_suffix = '_fold{fold}' if K > 1 else ''
	# This is also for plotting grouped but doesn't need to be commented out
	label_suffix = f'-{labels[0]}' if GROUP_BY else ''
	
	font_size = 16 if SIZE == 'large' else 32
	legend_size = 16 if SIZE == 'large' else 24
	size_suffix = '-large' if SIZE == 'large' else ''

	painter = None
	if PLOT:
		painter = Painter(
			lim=(0, 1.01),
			xticks=np.linspace(0.2, 1, 5),
			yticks=np.linspace(0, 1, 6),
			colors=['r', 'b', 'g', 'purple', 'black'],
			k=(len(test) if GROUP_BY else 1) * K * len(models),
			# For grouped
			#labels=(
			#	[f"{key} (k = {k})" for key in test.keys() for k in range(K)] if GROUP_BY and K > 1
			#	else [f"{key}" for key in test.keys()] if GROUP_BY
			#	else [f"k = {k}" for k in range(K)]
			#),
			# Otherwise
			labels=labels,
			font='Times New Roman',
			font_size=font_size,
			legend_size=legend_size,
			pause_on_end=False
		)
		painter.init()
		painter.add_figure('EER', xlabel='Threshold', ylabel='FAR/FRR')
		painter.add_figure(
			'ROC Curve',
			save=os.path.join(res_dir, f'Sclera-ROC{label_suffix}{size_suffix}.eps'),
			xlabel='FAR', ylabel='VER', legend_loc='lower right'
		)
		painter.add_figure(
			'Semilog ROC Curve',
			save=os.path.join(res_dir, f'Sclera-ROC-log{label_suffix}{size_suffix}.eps'),
			xlabel='FAR', ylabel='VER', legend_loc='best',
			xscale='log', xlim=(1e-3, 1.01), xticks=(1e-3, 1e-2, 1e-1, 1), x_tick_formatter=exp_format
		)

	try:
		res_str = []
		for model, label in zip(models, labels):
			evaluation = CV(model)(
				train,
				test,
				K,
				base_split_n=BASE_DIRS,
				plot=painter,
				closest_only=True,
				intergroup_evaluation=INTERGROUP,
				save=os.path.join(res_dir, 'Distance Matrices', f'{label}{group_suffix}{fold_suffix}.pkl'),
				use_precomputed=LOAD
			)
			if GROUP_BY:
				res_str.append(f"{label}:\n\n" + "\n\n".join(f"{k}:\n{str(v)}" for k, v in evaluation.items()))
			else:
				res_str.append(f"{label}:\n\n{str(evaluation)}")
			print(f"\n{'-' * 40}\n")
			print(res_str[-1])
			print(f"\n{'-' * 40}\n")
		if SAVE:
			with open(os.path.join(res_dir, f'Evaluation.txt'), 'w') as res_file:
				res_file.write("\n\n\n\n".join(res_str))
				res_file.write("\n")


	finally:
		if PLOT:
			painter.finalize()


# Configs
BATCH_SIZE = 32
DIST = 'cosine'


def base_nn_config(model, train=False, *args, feature_size=None, first_unfreeze=None, **kw):
	if not train:
		return PredictorModel(model, *args, batch_size=BATCH_SIZE, distance=DIST, **kw)
	return TrainablePredictorModel(
		model,
		*args,
		primary_epochs=100,
		secondary_epochs=50,
		feature_size=feature_size,
		first_unfreeze=first_unfreeze,
		primary_opt=RMSprop(lr=1e-4),
		secondary_opt=SGD(lr=1e-5, momentum=0.5, nesterov=True),
		batch_size=BATCH_SIZE,
		distance=DIST,
		**kw
	)


def resnet50(*args, **kw):
	return base_nn_config(ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg'), *args, first_unfreeze=143, **kw)


def scleranet(layer='final_features', image_size=None, *args, **kw):
	model = load_model(os.path.join(get_eyez_dir(), 'Recognition', 'Models', 'Rot ScleraNet', 'id_dir_prediction.75-0.667.hdf5'))
	if image_size and image_size != (400, 400):
		model.layers.pop(0)
		model = Model(model.input, model.get_layer(layer).output)
		input_ = Input(shape=(*image_size, 3))
		model = Model(input_, model(input_))
		return base_nn_config(model, *args, input_size=image_size, **kw)
	return base_nn_config(Model(model.input, model.get_layer(layer).output), *args, **kw)


def descriptor(*args, **kw):
	return DirectDistanceModel(DescriptorModel(*args, **kw))


def hog(*args, **kw):
	return PredictorModel(HOGModel(*args, **kw), batch_size=BATCH_SIZE, distance=DIST)


# Too slow and doesn't work
def correlation():
	return DirectDistanceModel(CorrelationModel())


# Different segmentation results (input datasets)
'''
old_dd, old_d = DATA_DIR, DATA
DATA_DIR = os.path.join(get_eyez_dir(), 'Segmentation', 'Results', 'Vessels')
for data_dir in ('Miura_MC_norm', 'Miura_RLT_norm', 'Coye', 'B-COSFIRE'):
	print(data_dir)
	DATA = {'train': None, 'test': data_dir}
	main()
DATA_DIR, DATA = old_dd, old_d
'''

# Different input resolutions
'''
old_dd, old_d, old_is = DATA_DIR, DATA, IMG_SIZE
DATA_DIR = os.path.join(get_eyez_dir(), 'Recognition', 'Databases', 'Rot ScleraNet')
for resolution in (400, 256, 192, 128, 96, 64):
	print(resolution)
	DATA = {'train': None, 'test': f'stage2_{resolution}x{resolution}' if resolution < 400 else 'stage2'}
	IMG_SIZE = (resolution, resolution)
	main()
DATA_DIR, DATA, IMG_SIZE = old_dd, old_d, old_is
'''


# Different grouping protocols
'''
old_gb, old_b, old_ig, old_i = GROUP_BY, BINS, INTERGROUP, I
for GROUP_BY, BINS in (('age', (25, 40)), ('gender', None)):
	for INTERGROUP in (False, True):
		for I in range(5):
			print(GROUP_BY, INTERGROUP, I)
			main()
GROUP_BY, BINS, INTERGROUP, I = old_gb, old_b, old_ig, old_i
'''

# Different base sizes
'''
old_bd = BASE_DIRS
for BASE_DIRS in range(1, 4):
	print(BASE_DIRS, "directions in base")
	main()
print("Center in base")
BASE_DIRS = (C,)
main()
BASE_DIRS = old_bd
'''


# Single run
main()
