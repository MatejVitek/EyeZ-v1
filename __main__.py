import os

from keras.applications import ResNet50

from cross_validate import CV


DATA_DIR = os.path.expanduser(os.path.join('~', 'EyeZ', 'Lozej', 'Datasets', 'Recognition'))
DATASET = 'SBVPI'
DATA = ('train', 'final_validation')


def main():
	dir = os.path.join(DATA_DIR, DATASET)
	model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	CV(
		model,
		os.path.join(dir, DATA[0]),
		os.path.join(dir, DATA[1]),
		first_unfreeze=143,
		primary_epochs=2,
		secondary_epochs=2,
		both_eyes_same_class=True,
		plot=True
	)(
		k=10,
		gp_split=0.3
	)


main()

