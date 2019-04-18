from functools import partial
import matplotlib.pyplot as plt
import os
import sys

from keras import backend
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.optimizers import RMSprop, SGD
from keras.utils.generic_utils import CustomObjectScope

from dataset import Dataset
from image_generators import ImageTupleGenerator
from naming import NamingParser
from utils import get_eyez_dir


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

	t_data = Dataset(os.path.join(get_eyez_dir(), 'Segmentation', 'Databases', 'Sclera', 'SSBC2019 + SBVPI', 'train', 'Images'), naming=naming)
	t_masks = Dataset(os.path.join(get_eyez_dir(), 'Segmentation', 'Databases', 'Sclera', 'SSBC2019 + SBVPI', 'train', 'Masks'), naming=naming)
	v_data = Dataset(os.path.join(get_eyez_dir(), 'Segmentation', 'Databases', 'Sclera', 'SSBC2019 + SBVPI', 'val', 'Images'), naming=naming)
	v_masks = Dataset(os.path.join(get_eyez_dir(), 'Segmentation', 'Databases', 'Sclera', 'SSBC2019 + SBVPI', 'val', 'Masks'), naming=naming)
	t_gen = ImageTupleGenerator((t_data, t_masks), target_size=(400, 400), color_mode=('rgb', 'grayscale'), flatten=(False, True), batch_size=BATCH_SIZE)
	v_gen = ImageTupleGenerator((v_data, v_masks), target_size=(400, 400), color_mode=('rgb', 'grayscale'), flatten=(False, True), batch_size=BATCH_SIZE)

	model = segnet(input_shape=(400, 400, 3), n_labels=1)
	cp_path = os.path.join(get_eyez_dir(), 'Segmentation', 'Models', 'Sclera', 'SegNet', 'ssbc2019+sbvpi_{epoch:03d}-{val_acc:.3f}.hdf5')
	_fit_model(model, t_gen, v_gen, 200, opt='SGD', loss=_categorical_crossentropy, cp_path=cp_path, plot=True)


def _fit_model(model, t_gen, v_gen, epochs, opt='SGD', loss='categorical_crossentropy', cp_path=None, plot=False):
	cb = [ModelCheckpoint(os.path.join('/tmp', 'best.h5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)]
	if cp_path:
		os.makedirs(os.path.dirname(cp_path), exist_ok=True)
		cb.append(ModelCheckpoint(cp_path, monitor='val_loss', verbose=1, save_best_only=True))
	if plot:
		plt.ion()
		cb.append(LambdaCallback(on_epoch_end=partial(_plot_prediction, model, v_gen)))
	model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
	model.fit_generator(
		t_gen,
		epochs=epochs,
		validation_data=v_gen,
		callbacks=cb
	)
	if plot:
		plt.ioff()
		plt.show()

	model.load_weights(os.path.join('/tmp', 'best.h5'))
	os.remove(os.path.join('/tmp', 'best.h5'))
	return model


# Function to display the target and prediciton
def _plot_prediction(model, v_gen, epoch, _):
	img, gt = next(v_gen)
	pred = model.predict(img).reshape(400, 400)
	plt.figure(f"Prediction {epoch}")
	plt.imshow(pred)
	plt.figure(f"GT {epoch}")
	plt.imshow(gt)


def _categorical_crossentropy(y_true, y_pred):
	eps = 1e-5
	y_pred = backend.clip(y_pred, eps, 1 - eps)
	return -backend.mean(y_true * backend.log(y_pred) + (1 - y_true) * backend.log(1 - y_pred))


# Configs
BATCH_SIZE = 4
DIST = 'cosine'


def deeplab(*args, **kw):
	sys.path.append(os.path.join(get_eyez_dir(), 'Recognition', 'Models', 'Lozej DeepLab+Xception'))
	from deeplab import relu6, BilinearUpsampling
	from initCombined import initModel
	with CustomObjectScope({'relu6': relu6, 'BilinearUpsampling': BilinearUpsampling}):
		model = initModel()
	return Model(model.input, model.get_layer('DeepLabOutput').output)


def segnet(*args, **kw):
	from SegNet.model import segnet
	return segnet(*args, **kw)


# Different segmentation results (input datasets)
#DATA_DIR = os.path.join(get_eyez_dir(), 'Segmentation', 'Results', 'Vessels')
#for data_dir in ('Coye', 'B-COSFIRE'):
#	DATA = {'train': None, 'test': data_dir}
#	main()

# Different input resolutions
#DATA_DIR = os.path.join(get_eyez_dir(), 'Recognition', 'Databases', 'Rot ScleraNet')
#for resolution in (400, 256, 192, 128, 96, 64):
#	print(resolution)
#	DATA = {'train': None, 'test': f'stage2_{resolution}x{resolution}' if resolution < 400 else 'stage2'}
#	IMG_SIZE = (resolution, resolution)
#	main()


# Different grouping protocols
#for grp, bins in (('age', (25, 40)), ('gender', None)):
#	GROUP_BY = grp
#	BINS = bins
#	for intergrp in (False, True):
#		INTERGROUP = intergrp
#		for I in range(5):
#			main()


# Different base sizes
#for n in range(1, 4):
#	BASE_DIRS = n
#	main()
#BASE_DIRS = (C,)
#main()


# Single run
main()
